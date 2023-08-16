import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from photocatalysis.deeplearning.helpers import one_hot_to_smile

print(f'CUDA GPU Available: {torch.cuda.is_available()}')

class VAE(nn.Module):
    def __init__(self, INPUT_SIZE=120, CHARSET_LEN=33, LATENT_DIM=292, kernel_sizes=(9,9,11)):
        super(VAE, self).__init__()
        self.INPUT_SIZE = INPUT_SIZE
        self.CHARSET_LEN = CHARSET_LEN
        self.LATENT_DIM = LATENT_DIM

        ### ENCODING
        # Convolutional Layers
        self.conv_1 = nn.Conv1d(self.INPUT_SIZE, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)

        # Fully Connected Layer
        self.linear_0 = nn.Linear(70, 435)

        # Mean and Variance Latent Layers
        self.mean_linear_1 = nn.Linear(435, self.LATENT_DIM)
        self.var_linear_2 = nn.Linear(435, self.LATENT_DIM)
        
        ### DECODING
        # Fully connected, GRU RNN, Fully connected layers
        # 3 sequential GRUs of hidden size 501. batch_first = True implies batch_dim first. 
        # Then, inputs into GRU are of shape [batch_size, seq_length (INPUT_SIZE, 120), Hin (LATENT_DIM, 292)]
        self.linear_3 = nn.Linear(self.LATENT_DIM, self.LATENT_DIM)
        self.stacked_gru = nn.GRU(self.LATENT_DIM, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, self.CHARSET_LEN)
        
        ### ACTIVATION and OUTPUT 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        # Convolutional
        print(x.shape)
        x = self.relu(self.conv_1(x))
        print(x.shape)
        x = self.relu(self.conv_2(x))
        print(x.shape)
        x = self.relu(self.conv_3(x))
        print(x.shape)

        # Flatten the Convultional output [batch_size, 10, 70] to make an input [batch_size, 10*7] for a fully connected layer
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))

        # Mean and logvariance latent vectors [batch_size, latent_dim]
        m, v = self.mean_linear_1(x), self.var_linear_2(x) 
        return m, v

    def reparameterize(self, mu_z, logvar_z):
        ## Sample a latent vector 'z', given its mean and std vectors
        # z ~ N(mu, std), is non-differentiable. While z ~ mu + eps (dot) std, where eps ~ N(0, 1), is differentiable. Why?
        # Since mu and std are now deterministic model outputs that can be trained by backprop, while the 'randomness' implicitly enters via the standard normal error/epsilon term
        gamma = 1e-2 # not sure why this is here...?
        epsilon = gamma * torch.randn_like(logvar_z) # 0 mean, unit variance noise of shape z_logvar
        std = torch.exp(0.5 * logvar_z)
        z = mu_z + epsilon * std
        return z

    def decode(self, z):
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

        # By applying a linear transformation independently to each element, the model has the flexibility to learn different weights for different features at different time steps.
        # These weights can capture complex relationships within each time step's output, such as identifying important features or capturing patterns specific to that moment.
        # We then reshape back to regain the sequence structure...
        out_independent = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_independent), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        xhat = self.decode(z)
        return xhat, mu_z, logvar_z
    

def variational_loss(x, reconstructed_x_mean, mu_z, logvar_z):
    # VAE reconstruction + KL-div loss
    BCE = F.binary_cross_entropy(reconstructed_x_mean, x, reduction='sum') # Pixel-wise reconstruction loss, no-mean taken to match KL-div
    KLD = -0.5 * torch.sum(1. + logvar_z - mu_z.pow(2) - logvar_z.exp()) # KL divergence of the latent space distribution

    return BCE + KLD, BCE, KLD

def train_epoch(training_data_loader, MODEL, OPTIMIZER, validation_data_loader=None, epoch=0, device='cpu', charset=None):
    print(f'################ epoch {epoch} ################')
    print('TRAINING')
    start = time.perf_counter()
    MODEL.train() # Tell model we're in train mode, as opposed to eval mode
    training_loss, training_bce_loss, training_kld_loss, validation_loss = 0, 0, 0, 0

    for batch_indx, X in enumerate(tqdm(training_data_loader)):
        # Reset gradients after each batch and send data to GPU if availables
        OPTIMIZER.zero_grad()
        X = X[0].to(device)

        # Forward pass through the model
        Xhat, mu_z, logvar_z = MODEL(X)

        # Determine Loss, perform backward pass, and update weights
        loss, bceloss, kldloss = variational_loss(X, Xhat, mu_z, logvar_z)
        loss.backward()
        OPTIMIZER.step()

        training_loss += loss.item()
        training_bce_loss += bceloss.item()
        training_kld_loss += kldloss.item()

    # Free up some memory on the GPU
    # del X
    # torch.cuda.empty_cache()

    if validation_data_loader is not None:
        MODEL.eval() # now we're in evaluation mode

        # Get model performance on validation set
        # Batch is too large to be stored on the GPU... need to break it up
        print('VALIDATING')
        for _, X_valid in enumerate(tqdm(validation_data_loader)):
            X_valid = X_valid.to(device)
            Xhat_valid, mu_valid, logvar_valid = MODEL(X_valid)
            validation_loss_batch, _, _ = variational_loss(X_valid, Xhat_valid, mu_valid, logvar_valid)
            validation_loss += validation_loss_batch.item()

        # X_valid = data_valid_tensor.to(device)
        # Xhat_valid, mu_valid, logvar_valid = model(X_valid)
        # validation_loss, _, _ = variational_loss(X_valid, Xhat_valid, mu_valid, logvar_valid)

        mean_validation_loss = validation_loss / len(validation_data_loader.dataset)
    else:
        mean_validation_loss = None

    # Summary of training epoch
    N = len(training_data_loader.dataset)
    mean_training_loss = training_loss / N
    mean_training_bce_loss = training_bce_loss / N
    mean_training_kld_loss = training_kld_loss / N

    test_points = X[0].cpu(), Xhat[0].cpu().detach() # Access a datapoint, send to cpu, and remove gradient
    test_smiles = [one_hot_to_smile(t.numpy(), charset) for t in test_points]

    print('SUMARRY')
    print(f'Epoch took: {(time.perf_counter() - start) / 60.} mins')
    print('Mean Training Loss:', mean_training_loss)
    print('Mean Validation Loss:', mean_validation_loss)
    print('---------------------')
    print('Random Sampled Input, Ouput Smiles:')
    for t in test_smiles: print(t)

    return (mean_training_loss, mean_training_bce_loss, mean_training_kld_loss), mean_validation_loss