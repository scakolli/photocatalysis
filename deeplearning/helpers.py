import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
import selfies as sf

import torch
from tqdm import tqdm

def get_charset(smiles_list, sos_token=None, eos_token=' '):
    char_list = list()
    max_smi_len = 0

    for smi in smiles_list:
        smi_len = len(smi)

        # Update maximum length smile
        if smi_len > max_smi_len:
            max_smi_len = smi_len
        
        # Capture unique characters
        for c in smi:
            if c not in char_list:
                char_list.append(c)

    # Prepending 'space' padding character and SOS token if provided
    if sos_token is not None: char_list.insert(0, 'X')
    char_list.insert(0, eos_token)

    return np.array(char_list), max_smi_len

def smiles_to_onehot(smiles_list, character_list, pad_length):
    character_list_ = list(character_list)
    oh_tensor = []
    
    for smi in smiles_list:
        # Pad smi with spaces
        padding = ' ' * (pad_length - len(smi))
        padded_smi = smi + padding
        smi_oh = []
        for c in padded_smi:
            oh = [0] * len(character_list_)
            indx = character_list_.index(c)
            # indx = np.where(character_list == c)[0][0]
            oh[indx] = 1
            smi_oh.append(oh)
        
        oh_tensor.append(smi_oh)

    return np.array(oh_tensor).astype(np.float32)

def one_hot_to_smile(onehot_vector, character_set):
    ### Take a one-hot vector/tensor (MAX SMILE LENGTH, CHARSET LENGTH) and convert it to a smile string (or selfie string)
    assert onehot_vector.shape[1] == character_set.size, 'Onehot length doesnt match character_set length'
    indicies = np.argmax(onehot_vector, axis=1)
    # return b''.join(character_set[indicies])
    return ''.join(character_set[indicies])

def plot_model_performance(tls, vls, name=None):
    trainlosses = pd.DataFrame(tls, columns=['train', 'bce', 'kld', 'prop'])
    validlosses = pd.DataFrame(vls, columns=['valid', 'bce', 'kld', 'prop'])

    trainlosses['elbo'] = trainlosses.bce + trainlosses.kld
    validlosses['elbo'] = validlosses.bce + validlosses.kld
    trainlosses['kld_elbo_ratio'] = 100 * trainlosses.kld / trainlosses.elbo # What % of the ELBO loss does KLD make up

    fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    trainlosses.train[1:].plot(label='train', color='red', ax=ax[0])
    validlosses.valid[1:].plot(label='valid', color='blue', ax=ax[0])
    ax[0].set_title('TOTAL')
    ax[0].legend()

    trainlosses.elbo[1:].plot(label='train', color='red', ax=ax[1])
    validlosses.elbo[1:].plot(label='valid', color='blue', ax=ax[1])
    ax[1].set_title('ELBO')
    ax[1].legend()

    trainlosses.kld_elbo_ratio[1:].plot(label='KLD Fraction of ELBO', color='green', ax=ax[2])
    ax[2].set_title('KLD / ELBO')

    trainlosses.prop[1:].plot(label='train', color='red', ax=ax[3])
    validlosses.prop[1:].plot(label='valid', color='blue', ax=ax[3])
    ax[3].set_title('PROP')
    ax[3].legend()

    plt.suptitle(name)

    return trainlosses, validlosses

####### Verifying Molecules ##################

def balanced_parentheses(smile):
    # Parentheses should be balance in the smile string
    s = []
    balanced = True
    index = 0
    while index < len(smile) and balanced:
        token = smile[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()

        index += 1

    return balanced and len(s) == 0

def matched_ring(smile):
    # Should be an even number of numbers, indicating proper ring closure
    return smile.count('1') % 2 == 0 and smile.count('2') % 2 == 0

def full_verify_smile(smile):
    return (smile != '') and pd.notnull(smile) and (Chem.MolFromSmiles(smile) is not None)

def fast_verify_smile(smile):
    return matched_ring(smile) and balanced_parentheses(smile)

def verify_smile(smile):
    # If initial fast verify passes, do thorough verify with rdkit, else return
    # False immediately
    if fast_verify_smile(smile):
        return full_verify_smile(smile)
    else:
        return False

def repeatative_verify_smile(smile, repeat_thresh=8):
    # Verify and filter out smiles with an excessive number of repeating non-carbon atoms
    nonC = ['O', 'S', 'N', 'o', 'n', 's']
    condition = [smile.count(atom) < repeat_thresh for atom in nonC]
    return all(condition)

######### SELFIES ############

def get_charset_selfies(selfies, sos_token=None, eos_token="[nop]"):
    
    max_selfie_len = max(sf.len_selfies(s) for s in selfies)
    alphabet = sf.get_alphabet_from_selfies(selfies)
    alphabet = list(sorted(alphabet))
    
    if sos_token is not None: alphabet.insert(0, sos_token)
    alphabet.insert(0, eos_token)  # [nop] no-operation special padding symbol

    # SOS-token is always in the 2nd pos; construct gotoken for TeacherForcing accordingly

    return np.array(alphabet), max_selfie_len

def selfies_to_onehot(selfies, alphabet, max_selfie_len=None):

   if max_selfie_len is not None:
      # Use user inputed sequence length
      pad_to_len = max_selfie_len
   else:
      # Use max sequence length encountered in selfies
      pad_to_len = max(sf.len_selfies(s) for s in selfies)

   symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

   one_hot_tensor = []
   for s in selfies:
      _, one_hot = sf.selfies_to_encoding(
         selfies=s,
         vocab_stoi=symbol_to_idx,
         pad_to_len=pad_to_len,
         enc_type="both")

      one_hot_tensor.append(one_hot)

   return np.array(one_hot_tensor).astype(np.float32)

# def onehot_to_selfies(one_hot_tensor, alphabet):
#     # Only works if 1's present in the tensor...wont work with raw logits
#     idx_to_symbol = {i: s for i, s in enumerate(alphabet)}

#     selfs = []
#     for oh in one_hot_tensor:
#         self = sf.encoding_to_selfies(oh, idx_to_symbol, enc_type="one_hot")
#         selfs.append(self)

#     return selfs

###### Performancy Metrics #######

def Accuracy(xhat, x):
    # Indices of highest score characters
    _, topi = x.topk(1)
    _, topi_hat = xhat.topk(1)
    # Xhat_onehot = torch.nn.functional.one_hot(topi_hat[:, :, 0]) # Xhat as a one-hot representation

    ### Character-by-character accuracy
    # Num equivalent chars / total chars
    eqv = topi == topi_hat
    char_acc = torch.sum(eqv) / topi.nelement()

    ### Smile accuracy
    # Num equivalent smiles / total smiles
    smi_acc = torch.all(eqv, dim=1).sum() / eqv.shape[0]

    return char_acc.item(), smi_acc.item()

def logits_batch_to_one_hot_batch(Logits_Batch):
    # Take a batch of logits (250, 80, 22) and sample a batch of onehot vectors multinomially (250, 80, 22)
    one_hots = []

    for l in Logits_Batch:
        indx = torch.multinomial(l, 1)
        one_hot = torch.zeros_like(l)
        one_hot.scatter_(1, indx, 1)
        one_hot.unsqueeze_(0)

        one_hots.append(one_hot)

    one_hots = torch.cat(one_hots, 0)

    return one_hots

def decode_logits(Logits, character_list, num_decode_attempts=100):
    # Probabilistic multinomial sampling of softmax output logist
    # and subsequent bare smile string decoding w/o validation
    assert Logits.shape[1] == character_list.size, 'Logit length doesnt match character_set length'
    decoded_smiles = []

    for _ in range(num_decode_attempts):
        indx = torch.multinomial(Logits, 1)
        one_hot = torch.zeros_like(Logits)
        one_hot.scatter_(1, indx, 1)
        
        smile = one_hot_to_smile(one_hot, character_list)
        decoded_smiles.append(smile)
    
    return decoded_smiles

def reconstruction_accuracy(loader, MODEL, num_encode_attempts=10, num_run_throughs=10, num_decode_attempts=10, num_mols=250, device='cpu'):
    ca, sa = [], []
    loader_batch_size = loader.batch_size

    for batch_indx, (xx, yy) in enumerate(loader):
        print(f'########## BATCH {batch_indx} ##########')
        
        with torch.no_grad():
            xx = xx.to(device)
            mu_z, logvar_z = MODEL.encode(xx)
            xx = xx.cpu()
            
            for i in tqdm(range(num_encode_attempts)):
                # 1. sample latent vector
                z = MODEL.reparameterize(mu_z, logvar_z)

                for j in range(num_run_throughs):
                    # 2. sample model run through
                    logits_batch, _ = MODEL.decode(z, groundTruth=None)
                    logits_batch = logits_batch.cpu()
                    
                    for k in range(num_decode_attempts):
                        # 3. sample multinomially a one-hot vector from logits
                        xxhat = logits_batch_to_one_hot_batch(logits_batch)

                        # Calculate Accuracy metrics of this sample
                        char_acc, smi_acc = Accuracy(xxhat, xx)
                        ca.append(char_acc), sa.append(smi_acc)
                        
        if batch_indx+1 == num_mols // loader_batch_size:
            break
    
    mean_char_acc, mean_smi_acc = np.mean(ca), np.mean(sa)

    trials = num_encode_attempts*num_run_throughs*num_decode_attempts
    print('#################')
    print(f'{trials} trials over {num_mols} molecules')
    print(f'Mean Character-by-Character Accuracy: {mean_char_acc}')
    print(f'Mean Smile Accuracy: {mean_smi_acc}')

    return mean_char_acc, mean_smi_acc

def prior_validity_accuracy(MODEL, character_set, num_z_samples=10, num_run_throughs=10, num_decode_attempts=10, eps_std=1., device='cpu', selfies=False):

    # Sample latent space
    Zs = eps_std * torch.randn(num_z_samples, MODEL.LATENT_DIM)
    Zs = Zs.repeat(num_run_throughs, 1)

    # Create dataloader so you can fit stuff on the GPU if need be
    loader = torch.utils.data.DataLoader(Zs, batch_size=250, shuffle=True)

    decoded_smiles_list = []

    for z in tqdm(loader):
        with torch.no_grad():
            z = z.to(device)
            logits_batch, _ = MODEL.decode(z, groundTruth=None)
            logits_batch = logits_batch.cpu()

            for l in logits_batch:
                smiles = decode_logits(l, character_set, num_decode_attempts=num_decode_attempts)
                decoded_smiles_list += smiles

    err_cnt = 0
    if selfies:
        # Decode selfies to smiles first
        success_decode = []
        for s in decoded_smiles_list:
            try:
                sm_decoded = sf.decoder(s)
                success_decode.append(sm_decoded)
            except:
                err_cnt += 1
                pass

        decoded_smiles_list = success_decode

    valid_smiles = [s for s in decoded_smiles_list if verify_smile(s)]
        
    # Proprtion of valid smiles
    trials = num_run_throughs*num_decode_attempts - err_cnt
    prior_valid_acc = len(valid_smiles) / len(decoded_smiles_list)

    print('#################')
    print(f'{trials} trials over {num_z_samples} molecules')
    print(f'{len(valid_smiles)} / {len(decoded_smiles_list)} valid smiles encountered')
    print(f'Prior Valid Fraction: {prior_valid_acc}')

    return prior_valid_acc
