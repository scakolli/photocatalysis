import numpy as np

def get_charset(smiles_list):
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

    # Append 'space' padding character
    char_list.append(' ')

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