import os
import time
import sys
from copy import deepcopy

from photocatalysis.thermodynamics.helpers import get_batches
from photocatalysis.evaluate import evaluate_substrate_in_batches
# from photocatalysis.thermodynamics.helpers import get_logger


# from photocatalysis.evaluate import evaluate_substrate

########################## RUN DEFINITIONS ###################################

# def run_evaluation():
#     with open(path_system_to_calculate) as out:
#         smi = out.readlines()[0]

#     try:
#         ip, rdg, asites, rds, essi = evaluate_substrate(smi, CALC_PARAMS) # scratch_dir=SCRATCH_DIR)
#         os.system('echo "{} {} {}" >> results.txt'.format(smi, ip, rdg, asites, rds, essi))
#         os.system('echo "{} {} {}" >> ../results_calculations.txt'.format(smi, ip, rdg, asites, rds, essi))

#         success = True
#     except Exception as e:
#         success = False

#         print('Error/Timeout Encountered')
#         print(e)
#         os.system('echo "{}" >> errors.txt'.format(e))
#         os.chdir(path_system_to_calculate_results)
    
#     return success

########################## XTB/FOLDER PARAMETERS ###################################
# SCRIPT = '/home/btpq/bt308495/Thesis/photocatalysis/worker/calc_xtb.py'
CALC_PARAMS = {'gfn':2, 'acc':0.2, 'etemp':298.15, 'gbsa':'water'}
NBATCH = 50
SCRATCH_DIR = '/home/btpq/bt308495/Thesis/scratch'

########################## RUN ###################################
if __name__ == '__main__':

    folder_to_calculate = sys.argv[1]
    # folder_to_calculate = 'molecules_to_calculate'
    # folder_to_calculate = '/home/btpq/bt308495/Thesis/run/learner_results/molecules_to_calculate'
    # os.chdir('/home/btpq/bt308495/Thesis/run/learner_results/')

    basedir = os.getcwd() # Active learner runs in ~/../learner_results
    cif_directory = os.path.join(basedir, folder_to_calculate)
    results_directory = os.path.join(basedir, "{}_results".format(folder_to_calculate))
    # os.system("ln -s {} .".format(SCRIPT))

    assert os.path.isdir(cif_directory), 'worker_xtb.py script runs in learner_results/ and on molecules_to_calculate/ of the active learning workflow'

    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    content_resultsdir = [x.split("__")[0] for x in os.listdir(results_directory)] # list molecules that ran
    systems_cif_dir_to_calc = [x for x in os.listdir(cif_directory)] # list molecules to process
    systems_cif_dir_to_calc = list(set(systems_cif_dir_to_calc) - set(content_resultsdir)) # process what remains

    # Order files by creation date
    # systems_cif_dir_to_calc = order_fxn(systems_cif_dir_to_calc, folder_systems=cif_directory)
    # Order files by pandas index/number
    keys = [int(''.join(filter(str.isdigit, fname))) for fname in systems_cif_dir_to_calc]
    systems_cif_dir_to_calc = [d for _, d in sorted(zip(keys, systems_cif_dir_to_calc))]

    if not len(systems_cif_dir_to_calc):
        print("No systems to calculate")
        sys.exit(0)

    total_num_systems = len(systems_cif_dir_to_calc)

    batches = get_batches(systems_cif_dir_to_calc, NBATCH)

    start = time.perf_counter()

    for b_num, b in enumerate(batches):
        ref_dict = {}
        for system_to_calculate in b:
            path_system_to_calculate = os.path.join(cif_directory, system_to_calculate)
            path_system_to_calculate_results = os.path.join(results_directory, '{}__running'.format(system_to_calculate))
            os.mkdir(path_system_to_calculate_results)

            with open(path_system_to_calculate) as out:
                smi = out.readlines()[0]
            
            ref_dict[smi] = path_system_to_calculate_results
        

        print('############################')
        print(f'BATCH {b_num} / {len(batches)}')
        print(f'Molecules to process: {len(b)}')

        systems = deepcopy(list(ref_dict.keys()))
        properties, errors = evaluate_substrate_in_batches(systems, CALC_PARAMS, scratch_dir=SCRATCH_DIR, batch_number=b_num)

        ### Successful
        for smi, prop in properties:
            # Write properties
            ip, rdg, asites, rds, essi = prop
            new_results_path = ref_dict[smi].replace('__running', '__completed')
            os.system('echo "{} {} {} {} {} {}" >> {}/results_calculations.txt'.format(smi, ip, rdg, asites, rds, essi, results_directory))
            os.system("mv {} {}".format(ref_dict[smi], new_results_path))

        ### Errors
        for smi, e in errors:
            new_results_path = ref_dict[smi].replace('__running', '__fizzled')
            os.system("echo '{}' >> {}/errors.txt".format(e, ref_dict[smi]))
            os.system("mv {} {}".format(ref_dict[smi], new_results_path))

        print('Batch Evaluation Took:', time.perf_counter() - start)

        break