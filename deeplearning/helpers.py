import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
import selfies as sf

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