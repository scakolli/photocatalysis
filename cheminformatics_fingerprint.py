import numpy as np
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray, BulkTanimotoSimilarity, TanimotoSimilarity
from scipy.spatial.distance import squareform
import multiprocessing
from itertools import chain, repeat
from tqdm import tqdm

        
def get_hashed_morgan_count_fingerprint(mol, radius=2, size=2048):
    ''' Generate a fixed-length Morgan count-fingerprint '''

    fp = AllChem.GetHashedMorganFingerprint(mol, radius, size)
    arr = np.zeros((0,), dtype=np.int8)
    ConvertToNumpyArray(fp,arr)
    return np.array(arr, dtype=np.int64) 

def get_tanimoto_distmat(X1, X2=[], pairwise=False):
    ''' SINGLE CORE PROCESSING Fast wrapper to calculate the Soergel distance matrix '''

    sim_mat  =  []
    if not len(X2):
        for i,x in enumerate(tqdm(X1)):
            sims = BulkTanimotoSimilarity(x,list(X1[i+1:]))
            sim_mat.extend(sims)

        sim_mat = squareform(sim_mat)
        sim_mat = sim_mat+np.eye(sim_mat.shape[0],sim_mat.shape[1])
    else:
        if not pairwise:
            for i,x in enumerate(tqdm(X1)):
                sims = BulkTanimotoSimilarity(x,list(X2))
                sim_mat.append(sims)

            sim_mat=np.array(sim_mat)
        else:
            # Pairwise similarity of two vectors
            # TanimotoSimilarity(X1[i], X2[i]), for all i
            sim_mat = np.array([TanimotoSimilarity(x1, x2) for x1, x2 in zip(X1, X2)])

    return 1.-sim_mat

def BulkTanimotoSimilarity_worker(job):
    # one_mol, rest_mol = job
    return BulkTanimotoSimilarity(*job)

def get_tanimoto_distmat_multiprocessing(X1, X2=[], multiprocess=1):
    ''' MULTI CORE PROCESSING Fast wrapper to calculate the Soergel distance matrix '''
    # Must be called as a subprocess to ensure picklability of 'BulkTanimotoSimilarity'

    sim_mat  =  []
    if not len(X2):
        # Multiprocessing
        print(f'Constructing Tanimoto Similarity Matrix. {len(X1)} mols to do')
        jobs = [(x, X1[i+1:]) for i, x in enumerate(X1)]
        with multiprocessing.Pool(processes=multiprocess) as pool:
            out = pool.map(BulkTanimotoSimilarity_worker, jobs)
    
        sim_mat = list(chain(*out))
        sim_mat = squareform(sim_mat)
        sim_mat += np.eye(sim_mat.shape[0],sim_mat.shape[1])
    else:
        # Multiprocessing
        print(f'Constructing Tanimoto Similarity Matrix. {len(X1)} mols to do')
        jobs = list(zip(X1, repeat(X2)))
        with multiprocessing.Pool(processes=multiprocess) as pool:
            sim_mat = pool.map(BulkTanimotoSimilarity_worker, jobs)
            
        sim_mat = np.array(sim_mat)
    
    return 1.-sim_mat