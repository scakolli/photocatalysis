import numpy as np
import pandas as pd
import os
import subprocess
import time
# import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import sys
import pickle

from functools import partial
from itertools import repeat, chain
import multiprocessing
import pathos
# from concurrent.futures import ProcessPoolExecutor

# from photocatalysis.adsorption.tools import prepare_substrate_batch, prepare_substrate
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
# from scipy.spatial.distance import squareform
from photocatalysis.cheminformatics_fingerprint import get_tanimoto_distmat
from photocatalysis.learners_treesearch import get_ML_model
from sklearn.model_selection import train_test_split
# from photocatalysis.gpr_model import GPR_tanimoto, _run_gpr_fit_bayesian_opt
# from photocatalysis.learners_treesearch import get_unique_population, get_population_completed, generate_ml_vectors
from photocatalysis.cheminformatics_fingerprint import Wrapper

### For automatically reloading import modules... allows you to run changes to code in jupyter without having to reload
%load_ext autoreload
%autoreload 2
# **********************************************************************
df = pd.read_json('/home/btpq/bt308495/Thesis/DF_COMPLETE.json', orient='split')
df_training, df_test = train_test_split(df, test_size=0.95, random_state=42)

with open('X.pckl', 'rb') as f:
    X = pickle.load(f)

X1 = X[:4000].copy().tolist()
# Y = df_training[:3000].IP.values

# gpr = GPR_tanimoto(multiprocess=8)

# o = gpr.fit(X1, Y)

# start = time.perf_counter()
# gpr_ip, xtrain_ip, kip = get_ML_model(df_training, 'IP', multiprocess=8)
# print('IP Fitting Took:', time.perf_counter() - start)

# ********************************************************************
# from joblib import Parallel, delayed

# jobs = [(x, X1[j+1:]) for j, x in enumerate(X1)]
# wrapped_BulkTanimotoSimilarity = Wrapper("BulkTanimotoSimilarity", "rdkit.DataStructs.cDataStructs")

# start = time.perf_counter()
# print('MULTIPROCESSING w/ joblib')
# o = Parallel(n_jobs=32)(delayed(wrapped_BulkTanimotoSimilarity)(one_mol, rest_mol) for one_mol, rest_mol in jobs)
# print('MP TOOK:', time.perf_counter()-start)

# **********************************************************************

# def BulkTanimotoSimilarity_worker(job):
#     return BulkTanimotoSimilarity(*job)

jobs = [(x, X1[j+1:]) for j, x in enumerate(X1)]

# start = time.perf_counter()
# print('SERIALPROCESSING')
# osp = get_tanimoto_distmat(X1)
# print('SP TOOK:', time.perf_counter()-start)

start = time.perf_counter()
print('MULTIPROCESSING')
omp = get_tanimoto_distmat(X1, multiprocess=4)
print('MP TOOK:', time.perf_counter()-start)

# def BulkTanimotoSimilarity_worker_wrapped(job):
#     return wrapped_BulkTanimotoSimilarity(*job)

# wrapped_BulkTanimotoSimilarity = Wrapper("BulkTanimotoSimilarity", "rdkit.DataStructs.cDataStructs")
# start = time.perf_counter()
# print('MULTIPROCESSING')
# with multiprocessing.Pool(processes=4) as pool:
#     out = pool.map(BulkTanimotoSimilarity_worker_wrapped, jobs)
# merged = list(chain(*out))
# print('MP TOOK:', time.perf_counter()-start)


# start = time.perf_counter()
# print('MULTIPROCESSING')
# with multiprocessing.Pool(processes=4) as pool:
#     out = pool.starmap(BulkTanimotoSimilarity, jobs)
# merged = list(chain(*out))
# print('MP TOOK:', time.perf_counter()-start)

# start = time.perf_counter()
# print('MULTIPROCESSING')
# with multiprocessing.Pool(processes=4) as pool:
#     out = pool.map(BulkTanimotoSimilarity_worker, jobs)
# merged = list(chain(*out))
# print('MP TOOK:', time.perf_counter()-start)

# start = time.perf_counter()
# print('MULTIPROCESSING DILL')
# with ProcessPoolExecutor(max_workers=8) as pool:
#     outp = pool.map(BulkTanimotoSimilarity_worker, jobs)
# mergedp = list(chain(*out))
# print('MP DILL TOOK:', time.perf_counter()-start)

# start = time.perf_counter()
# merged2 = list(chain(*map(BulkTanimotoSimilarity_worker, jobs)))
# print('SP TOOK:', time.perf_counter()-start)

# **********************************************************************


# path = '/home/btpq/bt308495/Thesis/osc_discovery/data/df_chemical_space_chons_4rings.json'
# p = pd.read_json(path, orient='split')

# # Change dir
# # os.chdir('/home/scakolli/Thesis/osc_discovery/run')
# os.chdir('/home/btpq/bt308495/Thesis/run')
# # sys.path.insert(1, '/home/btpq/bt308495/Thesis/')

# calc_params = {'gfn':2, 'acc':0.2, 'etemp':298.15, 'strict':'', 'gbsa':'water'}

# rand_smi = np.random.randint(0, 315450, size=32)

# smiles_toproc = p.molecule_smiles.iloc[rand_smi].tolist()

# def worker_func(job):
#     # job: tuple( job_number, (molecule, runtype, keep_folder boolean, calc_kwargs_dictionary) )
#     # Unpack job and return single_run() worker that can be used in multiprocessing code
#     job_num, job_input = job
#     smi, calc_paramss = job_input
#     return prepare_substrate(smi, calc_paramss, job_number=job_num)

# print('************1***************')
# print(os.getcwd())
# start1 = time.perf_counter()
# jobs = list(enumerate(zip(smiles_toproc, repeat(calc_params))))
# with mp.Pool(32) as pool:
#     # test = pool.starmap(prepare_substrate, zip(smiles_toproc, repeat(calc_params)))
#     test = pool.map(worker_func, jobs)
# end1 = time.perf_counter()

# print('************2***************')
# print(os.getcwd())
# start2 = time.perf_counter()
# test2 = prepare_substrate_batch(smiles_toproc, calc_params)
# end2 = time.perf_counter()

# print('Multiproc 1 Took', end1 - start1)
# print('Multiproc 2 Took', end2 - start2)