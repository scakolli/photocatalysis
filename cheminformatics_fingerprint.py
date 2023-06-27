import numpy as np
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray, BulkTanimotoSimilarity
from scipy.spatial.distance import squareform

def get_hashed_morgan_count_fingerprint(mol, radius=2, size=2048):
    ''' Generate a fixed-length Morgan count-fingerprint '''

    fp = AllChem.GetHashedMorganFingerprint(mol, radius, size)
    arr = np.zeros((0,), dtype=np.int8)
    ConvertToNumpyArray(fp,arr)
    return np.array(arr, dtype=np.int64) 


def get_tanimoto_distmat(X1,X2=[]):
    ''' Fast wrapper to calculate the Soergel distance matrix '''
    sim_mat  =  []
    if not len(X2):
        for  i,x in enumerate(X1):
            sims  =  BulkTanimotoSimilarity(x,list(X1[i+1:]))
            sim_mat.extend(sims)
        sim_mat = squareform(sim_mat)
        sim_mat = sim_mat+np.eye(sim_mat.shape[0],sim_mat.shape[1])
    else:
        for i,x in enumerate(X1):
            sims = BulkTanimotoSimilarity(x,list(X2))
            sim_mat.append(sims)
        sim_mat=np.array(sim_mat)
    
    return 1.-sim_mat 
