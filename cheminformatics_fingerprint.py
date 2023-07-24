import numpy as np
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray, BulkTanimotoSimilarity
from scipy.spatial.distance import squareform
import multiprocessing
from itertools import chain, repeat

def get_hashed_morgan_count_fingerprint(mol, radius=2, size=2048):
    ''' Generate a fixed-length Morgan count-fingerprint '''

    fp = AllChem.GetHashedMorganFingerprint(mol, radius, size)
    arr = np.zeros((0,), dtype=np.int8)
    ConvertToNumpyArray(fp,arr)
    return np.array(arr, dtype=np.int64) 

def BulkTanimotoSimilarity_worker(job):
    one_mol, rest_mol = job
    return BulkTanimotoSimilarity(one_mol, rest_mol)

def get_tanimoto_distmat(X1, X2=[], multiprocess=1):
    ''' Fast wrapper to calculate the Soergel distance matrix '''
    if len(X1) < 1000:
        # Multiprocessing slow for small vectors, better to single process...
        multiprocess = 1

    sim_mat  =  []
    if not len(X2):
        if multiprocess == 1:
            for  i,x in enumerate(X1):
                sims  =  BulkTanimotoSimilarity(x,list(X1[i+1:]))
                sim_mat.extend(sims)
        else:
            # Multiprocessing
            jobs = [(x, X1[i+1:]) for i, x in enumerate(X1)]
            with multiprocessing.Pool(processes=multiprocess) as pool:
                out = pool.map(BulkTanimotoSimilarity_worker, jobs)
        
            sim_mat = list(chain(*out))
        
        sim_mat = squareform(sim_mat)
        sim_mat = sim_mat+np.eye(sim_mat.shape[0],sim_mat.shape[1])
    else:
        if multiprocess == 1:
            for i,x in enumerate(X1):
                sims = BulkTanimotoSimilarity(x,list(X2))
                sim_mat.append(sims)
        else:
            # Multiprocessing
            jobs = list(zip(X1, repeat(X2)))
            with multiprocessing.Pool(processes=multiprocess) as pool:
                sim_mat = pool.map(BulkTanimotoSimilarity_worker, jobs)
            
        sim_mat=np.array(sim_mat)
    
    return 1.-sim_mat