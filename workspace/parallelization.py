# from pebble import ProcessPool
# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
from photocatalysis.evaluate import evaluate_substrate_in_batches, evaluate_substrate
# from itertools import repeat
import time
# import sys
# import traceback
from copy import deepcopy
import numpy as np
import pandas as pd

############# Read in #############
path = '/home/btpq/bt308495/Thesis/osc_discovery/data/df_chemical_space_chons_4rings.json'
p = pd.read_json(path, orient='split')

rand_smi = np.random.randint(0, 315450, size=64)

# Smiles to process
smile_string_list = p.molecule_smiles.iloc[rand_smi].tolist()

scratch_dir = '/home/btpq/bt308495/Thesis/scratch'
calc_kwargs = {'gfn':2, 'acc':0.2, 'etemp':298.15, 'strict':'', 'gbsa':'water'}

################ SERIAL VS PARALLEL ########################

# Parallel
start_parallel = time.perf_counter()
id_props, props_errors = evaluate_substrate_in_batches(smile_string_list, calc_kwargs, scratch_dir=scratch_dir)
end_parallel = time.perf_counter()
print('Parallel Took:', end_parallel - start_parallel)

# Serial
start = time.perf_counter()
props_test = []
for j, smile in enumerate(smile_string_list):
    try:
        props_test.append(evaluate_substrate(smile, calc_kwargs, scratch_dir=scratch_dir))
    except:
        props_test.append(np.nan)
end = time.perf_counter()

print('########################')
print('Serial Took:', end - start)
print('Parallel Took:', end_parallel - start_parallel)

props = [prop for _, prop in id_props]
ips = [(p[0], pt[0]) for p, pt in zip(props, props_test)]
rdgs = [(p[1], pt[1]) for p, pt in zip(props, props_test)]


print('#############PROPS##############')
print(ips)
print('################')
print(rdgs)

################### MULTIPROCESSING ################################

# jobs = list(enumerate(zip(smile_string_list, repeat(calc_kwargs))))

# start = time.perf_counter()
# with ProcessPool(max_workers=4) as pool:
#     fut = pool.map(prepare_substrate_worker, jobs)

# print('pebble took', time.perf_counter()-start)

# #     # for res in fut.result():

# start = time.perf_counter()
# with multiprocessing.Pool(processes=4) as pool:
#     res = pool.map(prepare_substrate_worker, jobs)

# print('mp map took', time.perf_counter()-start)

# def multiprocessing_run_and_catch(multiprocessing_iterator):
#     # Improved error handling using multiprocessing.Pool.imap()
#     # Supply a multiprocessing iterator. Try to run jobs and catch errors. Return only successfully completed jobs, and caught errors.
#     results, errors = [], []
#     iteration = 0
#     while True:
#         try:
#             result = next(multiprocessing_iterator)
#             results.append(result)
#         except StopIteration:
#             break
#         except Exception as e:
#             errors.append((iteration, traceback.format_exc()))
#         iteration += 1

#     return results, errors

# start = time.perf_counter()
# with multiprocessing.Pool(processes=4) as pool:
#     iterator = pool.imap(prepare_substrate_worker, jobs)
#     output, output_errors = multiprocessing_run_and_catch(iterator)

# print('mp imap took', time.perf_counter()-start)