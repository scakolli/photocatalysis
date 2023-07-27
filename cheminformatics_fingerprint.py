import numpy as np
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray, BulkTanimotoSimilarity
from scipy.spatial.distance import squareform
import multiprocessing
# import pathos
from itertools import chain, repeat
from functools import partial
import importlib
from tqdm import tqdm

class Wrapper:
    """Wrapper for RDKIT functions that imports necessary libraries before any serialization occurs
    in multiprocessing"""
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module = importlib.import_module(module_name)

    @property
    def method(self):
        return getattr(self.module, self.method_name)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)
        
def get_hashed_morgan_count_fingerprint(mol, radius=2, size=2048):
    ''' Generate a fixed-length Morgan count-fingerprint '''

    fp = AllChem.GetHashedMorganFingerprint(mol, radius, size)
    arr = np.zeros((0,), dtype=np.int8)
    ConvertToNumpyArray(fp,arr)
    return np.array(arr, dtype=np.int64) 

def BulkTanimotoSimilarity_worker(job):
    # one_mol, rest_mol = job
    return BulkTanimotoSimilarity(*job)

# def BulkTanimotoSimilarity_worker_wrapped(job):
#     wrapped_BulkTanimotoSimilarity = Wrapper("BulkTanimotoSimilarity", "rdkit.DataStructs.cDataStructs")
#     return wrapped_BulkTanimotoSimilarity(*job)

def get_tanimoto_distmat(X1, X2=[], multiprocess=1):
    ''' Fast wrapper to calculate the Soergel distance matrix '''
    lenx1 = len(X1)
    lenx2 = len(X2)
    if lenx1 < 1000:
        # Multiprocessing slow for small vectors, better to single process...
        multiprocess = 1

    sim_mat  =  []
    if not lenx2:
        if multiprocess == 1:
            print(f'Constructing Tanimoto Kernel. {lenx1} mols to do')
            for i,x in enumerate(tqdm(X1)):
                sims = BulkTanimotoSimilarity(x,list(X1[i+1:]))
                sim_mat.extend(sims)

                # if i % 100 == 0:
                #     print(f'BulkTanimoto Iter: {i} / {lenx1}')
        else:
            # Multiprocessing
            print(f'Constructing Tanimoto Kernel. {lenx1} mols to do')
            jobs = [(x, X1[i+1:]) for i, x in enumerate(X1)]
            with multiprocessing.Pool(processes=multiprocess) as pool:
                out = pool.map(BulkTanimotoSimilarity_worker, jobs)

                # wrapped_BulkTanimotoSimilarity = Wrapper("BulkTanimotoSimilarity", "rdkit.DataStructs.cDataStructs")
                # out = pool.starmap(wrapped_BulkTanimotoSimilarity, jobs)
                # out = pool.starmap(BulkTanimotoSimilarity, jobs)

                # wrapped_BulkTanimotoSimilarity = Wrapper("BulkTanimotoSimilarity", "rdkit.DataStructs.cDataStructs")
                # out = pool.starmap(wrapped_BulkTanimotoSimilarity, jobs)
        
            sim_mat = list(chain(*out))
        
        sim_mat = squareform(sim_mat)
        sim_mat = sim_mat+np.eye(sim_mat.shape[0],sim_mat.shape[1])
    else:
        if multiprocess == 1:
            print(f'Constructing Tanimoto Kernel. {lenx1} mols to do')
            for i,x in enumerate(tqdm(X1)):
                sims = BulkTanimotoSimilarity(x,list(X2))
                sim_mat.append(sims)
        else:
            # Multiprocessing
            print(f'Constructing Tanimoto Kernel. {lenx1} mols to do')
            jobs = list(zip(X1, repeat(X2)))
            with multiprocessing.Pool(processes=multiprocess) as pool:
                sim_mat = pool.map(BulkTanimotoSimilarity_worker, jobs)
            
        sim_mat=np.array(sim_mat)
    
    return 1.-sim_mat