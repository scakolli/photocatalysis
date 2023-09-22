import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_charset(smiles_list, sos_token=None):
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
    char_list.insert(0, ' ')

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
    ### Take a one-hot vector/tensor (MAX SMILE LENGTH, CHARSET LENGTH) and convert it to a smile string
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