import numpy as np
import pandas as pd
import os
import subprocess
import time
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import sys
import multiprocessing as mp

from rdkit import Chem
from rdkit.Chem import AllChem
import ase
from ase.io import read, write
from ase.units import Hartree
from ase.visualize import view

from ase.io.trajectory import Trajectory
import glob
import pickle

from functools import partial
from itertools import repeat

from photocatalysis.adsorption.tools import prepare_substrate_batch, prepare_substrate


path = '/home/btpq/bt308495/Thesis/osc_discovery/data/df_chemical_space_chons_4rings.json'
p = pd.read_json(path, orient='split')

# Change dir
# os.chdir('/home/scakolli/Thesis/osc_discovery/run')
os.chdir('/home/btpq/bt308495/Thesis/run')
# sys.path.insert(1, '/home/btpq/bt308495/Thesis/')

calc_params = {'gfn':2, 'acc':0.2, 'etemp':298.15, 'strict':'', 'gbsa':'water'}

rand_smi = np.random.randint(0, 315450, size=32)

smiles_toproc = p.molecule_smiles.iloc[rand_smi].tolist()

def worker_func(job):
    # job: tuple( job_number, (molecule, runtype, keep_folder boolean, calc_kwargs_dictionary) )
    # Unpack job and return single_run() worker that can be used in multiprocessing code
    job_num, job_input = job
    smi, calc_paramss = job_input
    return prepare_substrate(smi, calc_paramss, job_number=job_num)

print('************1***************')
print(os.getcwd())
start1 = time.perf_counter()
jobs = list(enumerate(zip(smiles_toproc, repeat(calc_params))))
with mp.Pool(32) as pool:
    # test = pool.starmap(prepare_substrate, zip(smiles_toproc, repeat(calc_params)))
    test = pool.map(worker_func, jobs)
end1 = time.perf_counter()

print('************2***************')
print(os.getcwd())
start2 = time.perf_counter()
test2 = prepare_substrate_batch(smiles_toproc, calc_params)
end2 = time.perf_counter()

print('Multiproc 1 Took', end1 - start1)
print('Multiproc 2 Took', end2 - start2)