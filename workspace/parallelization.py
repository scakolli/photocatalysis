# from pebble import ProcessPool
# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
from photocatalysis.evaluate import evaluate_substrate_in_batches, evaluate_substrate
# from itertools import repeat
import time
import sys
import traceback
from copy import deepcopy

smile_string_list = ['C1=CC(c2cc(C=Cc3ncns3)cc(C3=CCC=C3)n2)=CC1',
 'C1=CCC(c2cc(C3=CC=CC3)cc(-c3cc[nH]c3)c2)=C1',
 'C1=CCC(c2ccnnc2-c2nnccc2C2=CC=CC2)=C1',
 'C(#Cc1cc(C#CC2=CCN=C2)cc(-c2cc[nH]n2)c1)C1=CCN=C1']
 'C1=CCC(C=Cc2cccc(C=CC3=CC=CC3)c2-c2ccsc2)=C1',
 'C1=CC(=C2C(C=Cc3ccncc3)=CC=C2C=Cc2ccncc2)N=N1',
 'O=c1[nH]c2ccoc2c2c1N=CC2=C1C=CC=C1',
 'O=C1Cc2n[nH]c3cc(=C4C=CC=C4)cc-3c2=N1']

nbatch = len(smile_string_list)
scratch_dir = '/home/btpq/bt308495/Thesis/scratch'
calc_kwargs = {'gfn':2, 'acc':0.2, 'etemp':298.15, 'strict':'', 'gbsa':'water'}

# Serial
start = time.perf_counter()
props_test = []
for smile in smile_string_list:
    props_test.append(evaluate_substrate(smile, calc_kwargs, scratch_dir=scratch_dir))
end = time.perf_counter()

# # Parallel
start_parallel = time.perf_counter()
id_props, props_errors = evaluate_substrate_in_batches(smile_string_list, calc_kwargs, scratch_dir=scratch_dir)
end_parallel = time.perf_counter()

print('Serial Took:', end - start)
print('Parallel Took:', end_parallel - start_parallel)

props = [prop for _, prop in id_props]
ips = [(p[0], pt[0]) for p, pt in zip(props, props_test)]
rdgs = [(p[1], pt[1]) for p, pt in zip(props, props_test)]

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