import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from photocatalysis.deeplearning.helpers import one_hot_to_smile

class advGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, activation=F.tanh, inner_activation=F.sigmoid):
        super(advGRUCell, self).__init__(input_size, hidden_size, bias=True)
        self.activation = activation
        self.inner_activation = inner_activation

    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.autograd.Variable(input.data.new(
                input.size(0),
                self.hidden_size).zero_(), requires_grad=False)
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)

        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        resetgate = self.inner_activation(i_r + h_r)
        inputgate = self.inner_activation(i_z + h_z)
        preactivation = i_n + resetgate * h_n
        newgate = self.activation(preactivation)
        hy = newgate + inputgate * (hx - newgate)

        return hy, preactivation

class teacherGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 gru_activation=F.tanh, gru_inner_activation=F.sigmoid,
                 gotoken=None, state_dict=None, probabilistic_sampling=True):
        
        if gotoken is None:
            raise ValueError("Need to provide a gotoken when using teachers forcing")
        
        super(teacherGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gotoken = gotoken
        
        self.cell = advGRUCell(input_size=input_size + output_size, hidden_size=hidden_size, 
                               activation=gru_activation, inner_activation=gru_inner_activation)

        self.linear = nn.Linear(hidden_size, output_size)

        if state_dict is not None:
            self.cell.load_state_dict(state_dict[0])
            self.linear.load_state_dict(state_dict[1])

        if probabilistic_sampling:
            self.sample = torch.multinomial
        else:
            def topi(matrix, top):
                return torch.topk(matrix, top)[1]
            self.sample = topi

    def forward(self, y, groundTruth=None, hx=None):
        batch_size = y.size(0)
        seq_length = y.size(1)

        output = []
        sampled_output = []
        preactivation = []

        target = self.gotoken.repeat(batch_size, 1)

        if hx is None:
            hx = y.data.new(batch_size, self.hidden_size).zero_()

        for i in range(seq_length):
            input_ = torch.cat([y[:, i, :], target], dim=-1)
            hx, pre = self.cell(input_, hx=hx)
            output_ = F.log_softmax(self.linear(hx), dim=1)
            
            # Sampling
            probs = torch.exp(output_)
            indices = self.sample(probs, 1)
            one_hot = output_.data.new(output_.size(0), self.output_size).zero_() # originally was self.hidden_size, although i think this is a mistake
            one_hot.scatter_(1, indices, 1)

            # Construct output lists
            output.append(probs.view(batch_size, 1, self.output_size))
            preactivation.append(pre.view(batch_size, 1, self.hidden_size))
            sampled_output.append(one_hot)

            if groundTruth is not None:
                # Teacher force actual ground-truth
                target = groundTruth[:, i, :]
            else:
                # Feed in own prediction
                target = one_hot
        
        output = torch.cat(output, 1) # log probabilites
        preactivation = torch.cat(preactivation, 1)
        sampled_output = torch.stack(sampled_output, 1)
        
        # output probabilities instead of log probs
        return output, preactivation, sampled_output, hx
    
class propertyPredictor(nn.Module):
    def __init__(self, latent_input_dim=292, hidden_size=1000, hidden_depth=1, inner_activation=nn.Tanh()):
        super(propertyPredictor, self).__init__()
        self.hidden_depth = hidden_depth

        self.linear_in = nn.Linear(latent_input_dim, hidden_size)
        self.linear_mid_i = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)
        self.inner_activation = inner_activation

        self.mid_module_list = nn.ModuleList()
        for _ in range(hidden_depth):
            # Linear fully connected layer followed by activation (nn.Tanh() or nn.SELU() for example)
            self.mid_module_list.extend([self.linear_mid_i, self.inner_activation])

        self.linear_mid = nn.Sequential(*self.mid_module_list)

    def forward(self, z):
        embedded_z = self.inner_activation(self.linear_in(z)) # Explicitly include activation
        hidden_z = self.linear_mid(embedded_z)
        yhat = self.linear_out(hidden_z)

        return yhat

class noTeachernoGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        # NO GRU implemented.... just the old terminal layer and softmax
        super(noTeachernoGRU, self).__init__()
        
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, output, hx=None, groundTruth=None):
        out_independent = output.contiguous().view(-1, output.size(-1))
        logits = self.linear(out_independent)
        y0 = F.softmax(logits, dim=1)
        xhat = y0.contiguous().view(output.size(0), -1, y0.size(-1))

        return xhat, None, None, None

class VAE(nn.Module):
    def __init__(self, INPUT_SIZE=120, CHARSET_LEN=33, LATENT_DIM=292, HIDDEN_SIZE=501,
                filter_sizes=(9,9,10), kernel_sizes=(9,9,11), eps_std=1.,
                useTeacher=True,gotoken=None, probabilistic_sampling=True,
                property_prediction_params_dict=None):
        
        if useTeacher and gotoken is None:
            raise ValueError("Need to provide a gotoken when using teachers forcing")
        
        super(VAE, self).__init__()
        self.INPUT_SIZE = INPUT_SIZE
        self.CHARSET_LEN = CHARSET_LEN
        self.LATENT_DIM = LATENT_DIM
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.FS1, self.FS2, self.FS3 = filter_sizes
        self.KS1, self.KS2, self.KS3 = kernel_sizes
        self.eps_std = eps_std
        self.generation_mode = False

        ### PROP PREDICTION
        if property_prediction_params_dict is not None:
            self.property_predictor = propertyPredictor(latent_input_dim=self.LATENT_DIM, **property_prediction_params_dict)
        else:
            def retNone(*args):
                return None
            self.property_predictor = retNone

        ### ENCODING
        # Convolutional Layers
        self.conv_1 = nn.Conv1d(self.INPUT_SIZE, self.FS1, kernel_size=self.KS1)
        self.conv_2 = nn.Conv1d(self.FS1, self.FS2, kernel_size=self.KS2)
        self.conv_3 = nn.Conv1d(self.FS2, self.FS3, kernel_size=self.KS3)

        # Fully Connected Layer
        self.inp_s = self.FS3 * (self.CHARSET_LEN - self.KS1 - self.KS2 - self.KS3 + 3) # conv3.shape[1] * conv3.shape[2]
        self.linear_0 = nn.Linear(self.inp_s, 435)

        # Mean and Variance Latent Layers
        self.mean_linear_1 = nn.Linear(435, self.LATENT_DIM)
        self.var_linear_2 = nn.Linear(435, self.LATENT_DIM)
        
        ### DECODING
        # Fully connected, GRU RNN, Fully connected layers
        # 3 sequential GRUs of hidden size 501. batch_first = True implies batch_dim first. 
        # Then, inputs into GRU are of shape [batch_size, seq_length (INPUT_SIZE, 120), Hin (LATENT_DIM, 292)]
        
        # Embedding latent vector
        self.linear_3 = nn.Linear(self.LATENT_DIM, self.LATENT_DIM)

        # Pass to GRU
        self.stacked_gru = nn.GRU(self.LATENT_DIM, self.HIDDEN_SIZE, 3, batch_first=True)

        # Terminal GRU
        if useTeacher:
            self.terminalGRU = teacherGRU(self.HIDDEN_SIZE, self.HIDDEN_SIZE, self.CHARSET_LEN,
                                          gotoken=gotoken, probabilistic_sampling=probabilistic_sampling)
        else:
            self.terminalGRU = noTeachernoGRU(self.HIDDEN_SIZE, self.CHARSET_LEN)

        # Project GRU output to CHARSET_LEN
        # self.linear_4 = nn.Linear(501, self.CHARSET_LEN)
        
        ### ACTIVATION and OUTPUT 
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def encode(self, x):
        # Convolutional
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))

        # Flatten the Convultional output [batch_size, 10, 70] to make an input [batch_size, 10*7] for a fully connected layer
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))

        # Mean and logvariance latent vectors [batch_size, latent_dim]
        m, logv = self.mean_linear_1(x), self.var_linear_2(x) 
        return m, logv

    def reparameterize(self, mu_z, logvar_z):
        ## Sample a latent vector 'z', given its mean and std vectors
        # z ~ N(mu, std), is non-differentiable. While z ~ mu + eps (dot) std, where eps ~ N(0, 1), is differentiable. Why?
        # Since mu and std are now deterministic model outputs that can be trained by backprop, while the 'randomness' implicitly enters via the standard normal error/epsilon term
        epsilon = self.eps_std * torch.randn_like(logvar_z) # 0 mean, unit variance noise of shape z_logvar
        std = torch.exp(0.5 * logvar_z)
        z = mu_z + epsilon * std
        return z
    
    def decode(self, z, hx=None, groundTruth=None):
        # Fully connected layer to GRU
        z_emb = F.selu(self.linear_3(z))
        z_emb = z_emb.view(z_emb.size(0), 1, z_emb.size(-1)).repeat(1, self.INPUT_SIZE, 1)

        # Stacked + teacher-forcing GRU
        output, hs = self.stacked_gru(z_emb)
        logits, _, sampled_output, _ = self.terminalGRU(output, hx=hx, groundTruth=groundTruth)

        return logits, sampled_output

    def decode_old(self, z):
        z = F.selu(self.linear_3(z))

        # Since the GRU, when unrolled in 'time', consists of 120 NNs each sequentially processing data... we have to send 120 copies through it.
        # By repeating the tensor z self.INPUT_SIZE times along the sequence length dimension, we are effectively creating a sequence of self.INPUT_SIZE time steps,
        # each with the same latent representation. This setup allows the GRU to process this "sequence" of repeated tensors, even though the actual sequence content
        # is the same at each time step. This kind of setup can be useful for example when:

        # 1. Information Propagation: 
        # Sometimes you want to ensure that a certain piece of information is propagated consistently through the entire sequence. By using repeated tensors, you can
        # ensure that the same information is available to the network at every time step, allowing the network to incorporate this information throughout the entire sequence.

        # 2. Fixed-Size Context: If you have a fixed-size context or control signal that should influence the processing of the entire sequence, you can repeat this
        # context along the sequence length dimension. This way, the network can take into account this context when making decisions at every time step.

        # Note on use of contiguous()
        # contiguous means 'sharing a common border; touching'
        # In the context of pytorch, contiguous means not only contiguous in memory (each element in a tensor is stored right next to the other, in a block),
        # but also in the same order in memory as the indices order. For example doing a transposition doesn't change the data in memory (data at (1, 4) doesnt swap
        # memory places when its transposed to (4, 1)), it simply changes the map from indices to memory pointers (what index corresponds to what data is swapped instead,
        # leaving memory untouched). If you then apply contiguous() it will change the data in memory so that the map from indices to memory location is the canonical one.
        # For certain pytorch operations, contiguously stored tensors are required! Else a runtime error is encountered (RuntimeError: input is not contiguous).

        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.INPUT_SIZE, 1) # Reshape z from [batch_size, latent_dim] to [batch_size, seq_len (120), latent_dim]
        output, hs = self.stacked_gru(z) # hs represents the hidden state of the last time step of the GRU

        # Output is flattened along 1st two dimensions [batch_size, seq_len, hout] -> [batch_size * seq_len, hout]
        # Softmax is then applied row-wise/sample-wise following a linear transform
        # before the vector is then unflatten back to the original [batch_size, seq_len, charset_len]

        # The purpose of this initial flattening is:
        # In the context of a sequence-to-sequence model, each time step's output from the RNN (or a similar sequential model) represents the model's understanding of the
        # data at that particular moment. When you collapse the dimensions and reshape the tensor to (batch_size * sequence_length, num_features), you effectively create 
        # a flat sequence where each element corresponds to a time step's output for a specific sample in the batch.
        # Then applying a linear transformation like self.linear_4 at this stage means that the same linear transformation is applied to each element in the flattened sequence
        # independently (as if the new batch size is of shape batch_size * seq_len)! This is independent in the sense that the transformation doesn't consider interactions
        # between different time steps or different samples within the batch. It's a per-element operation.

        # Project GRU output back into the charset space (501 -> 21)
        # Either done by flatten t
        out_independent = output.contiguous().view(-1, output.size(-1))
        logits = self.linear_4(out_independent)
        y0 = F.softmax(logits, dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        # Encode
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        # xhat = self.decode(z)

        # Property Prediction if param dict present, else None
        yhat = self.property_predictor(z)
        
        # Decode with Teacher-forcing when training, else use model sequence predictions
        ground_truth = x if not self.generation_mode else None
        xhat, samples = self.decode(z, groundTruth=ground_truth)
        return (xhat, samples, mu_z, logvar_z), yhat
    
def cyclical_annealing(step, T, M, R=0.5, max_kl_weight=1):
    """
    Implementing: <https://arxiv.org/abs/1903.10145>
    T = Total steps or training epochs
    M = Number of cycles 
    R = Proportion used to increase beta
    step = Global step 
    """
    period = T / M # N_iters/N_cycles 
    internal_period = step % period  # Itteration_number/(Global Period)
    beta = internal_period / period
    if beta > R:
        beta = max_kl_weight
    else:
        beta = min(max_kl_weight, beta / R) # Linear function 
    return beta

def linear_annealing(step, T, R=0.8, max_kl_weight=1.):
    return min(max_kl_weight, step / int(T * R))
    
def variational_loss(x, reconstructed_x_mean, mu_z, logvar_z, beta=1.):
    # VAE ELBO, reconstruction + KL-div loss
    BCE = F.binary_cross_entropy(reconstructed_x_mean, x, reduction='sum') # Pixel-wise reconstruction loss, no-mean taken to match KL-div
    KLD = -0.5 * torch.sum(1. + logvar_z - mu_z.pow(2) - logvar_z.exp()) # KL divergence of the latent space distribution
    ELBO = BCE + beta * KLD
    return ELBO, BCE, KLD

def property_loss(ytrue, yhat):
    return F.mse_loss(ytrue, yhat, reduction='sum')

def train_epoch(training_data_loader, MODEL, OPTIMIZER, validation_data_loader=None, epoch=0, device='cpu', charset=None, kl_weight=1., prop_loss_weight=1., eos_token=' '):
    print(f'################ epoch {epoch} ################')
    print('TRAINING')
    start = time.perf_counter()
    MODEL.train() # Tell model we're in train mode, as opposed to eval mode
    training_loss, training_bce_loss, training_kld_loss, training_prop_loss = 0., 0., 0., 0.
    validation_loss, validation_bce_loss, validation_kld_loss, validation_prop_loss = 0., 0., 0., 0.

    for batch_indx, (X, y) in enumerate(tqdm(training_data_loader)):
        # Reset gradients after each batch and send data to GPU if availables
        OPTIMIZER.zero_grad()

        # X = X[0].to(device) # if using torch.utils.data.TensorDataset()
        X = X.to(device)
        y = y.to(device)

        # Forward pass through the model
        (Xhat, samples, mu_z, logvar_z), yhat = MODEL(X)

        # Determine Loss, perform backward pass, and update weights
        vloss, bceloss, kldloss = variational_loss(X, Xhat, mu_z, logvar_z, beta=kl_weight)

        if yhat is not None:
            proploss = property_loss(y, yhat)
        else:
            proploss = torch.tensor(0.)

        loss = vloss + proploss

        loss.backward()
        OPTIMIZER.step()

        training_loss += loss.item()
        training_bce_loss += bceloss.item()
        training_kld_loss += kldloss.item()
        training_prop_loss += proploss.item()

    # Free up some memory on the GPU
    # del X
    # torch.cuda.empty_cache()

    if validation_data_loader is not None:
        MODEL.eval() # now we're in evaluation mode

        # Get model performance on validation set
        # Batch is too large to be stored on the GPU... need to break it up
        print('VALIDATING')
        for _, (X_valid, y_valid) in enumerate(tqdm(validation_data_loader)):
            X_valid = X_valid.to(device)
            y_valid = y_valid.to(device)

            (Xhat_valid, samples_valid, mu_valid, logvar_valid), yhat_valid = MODEL(X_valid)
            vloss, bceloss, kldloss = variational_loss(X_valid, Xhat_valid, mu_valid, logvar_valid)

            if yhat is not None:
                proploss = property_loss(y_valid, yhat_valid)
            else:
                proploss = torch.tensor(0.)

            loss = vloss + proploss

            validation_loss += loss.item()
            validation_bce_loss += bceloss.item()
            validation_kld_loss += kldloss.item()
            validation_prop_loss += proploss.item()

        # X_valid = data_valid_tensor.to(device)
        # Xhat_valid, mu_valid, logvar_valid = model(X_valid)
        # validation_loss, _, _ = variational_loss(X_valid, Xhat_valid, mu_valid, logvar_valid)

        N = len(validation_data_loader.dataset)
        mean_validation_loss = validation_loss / N
        mean_validation_bce_loss = validation_bce_loss / N
        mean_validation_kld_loss = validation_kld_loss / N
        mean_validation_prop_loss = validation_prop_loss / N
    else:
        mean_validation_loss, mean_validation_prop_loss, mean_validation_bce_loss, mean_validation_kld_loss = None, None, None, None

    # Summary of training epoch
    N = len(training_data_loader.dataset)
    mean_training_loss = training_loss / N
    mean_training_bce_loss = training_bce_loss / N
    mean_training_kld_loss = training_kld_loss / N
    mean_training_prop_loss = training_prop_loss / N

    all_train_losses = (mean_training_loss, mean_training_bce_loss, mean_training_kld_loss, mean_training_prop_loss)
    all_valid_losses = (mean_validation_loss, mean_validation_bce_loss, mean_validation_kld_loss, mean_validation_prop_loss)

    test_points = X[0].cpu(), Xhat[0].cpu().detach() # Access a datapoint, send to cpu, and remove gradient
    test_smiles = [one_hot_to_smile(t.numpy(), charset).replace(eos_token, '') for t in test_points]

    print('EPOCH SUMARRY')
    print(f'Epoch took: {(time.perf_counter() - start) / 60.} mins')
    print('Mean Training BCE Loss:', mean_training_bce_loss)
    print('Mean Training KLD Loss:', mean_training_kld_loss)
    print('Mean Training Prop Loss:', mean_training_prop_loss)
    print('Mean Training ELBO Loss:', mean_training_bce_loss + mean_training_kld_loss)
    print('---------------------')
    print('Mean Training Loss:', mean_training_loss)
    print('Mean Validation Loss:', mean_validation_loss)
    print('---------------------')
    print('Random Sampled Input, Ouput Smiles:')
    for t in test_smiles: print(t)

    return all_train_losses, all_valid_losses