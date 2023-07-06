import os
import subprocess
import time
import sys

from photocatalysis.thermodynamics.helpers import get_batches
from photocatalysis.adsorption.tools import prepare_substrate_batch, multi_prepare_substrate
from photocatalysis.adsorption.relaxing import build_and_relax_configurations
from photocatalysis.evaluate import calculate_thermochemistry, global_min_configurations

# from photocatalysis.evaluate import evaluate_substrate

########################## RUN DEFINITIONS ###################################

def run_evaluation():
    with open(path_system_to_calculate) as out:
        smi = out.readlines()[0]

    try:
        ip, rdg, asites, rds, essi = evaluate_substrate(smi, CALC_PARAMS) # scratch_dir=SCRATCH_DIR)
        os.system('echo "{} {} {}" >> results.txt'.format(smi, ip, rdg, asites, rds, essi))
        os.system('echo "{} {} {}" >> ../results_calculations.txt'.format(smi, ip, rdg, asites, rds, essi))

        success = True
    except Exception as e:
        success = False

        print('Error/Timeout Encountered')
        print(e)
        os.system('echo "{}" >> errors.txt'.format(e))
        os.chdir(path_system_to_calculate_results)
    
    return success

########################## XTB/FOLDER PARAMETERS ###################################
# SCRIPT = '/home/btpq/bt308495/Thesis/photocatalysis/worker/calc_xtb.py'
CALC_PARAMS = {'gfn':2, 'acc':0.2, 'etemp':298.15, 'gbsa':'water'}
NBATCH = 50
# SCRATCH_DIR = '/home/btpq/bt308495/Thesis/scratch'

########################## RUN ###################################
if __name__ == '__main__':
    
    folder_to_calculate = sys.argv[1]

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

    print('#################')
    batches = get_batches(systems_cif_dir_to_calc, NBATCH)

    eval_logger = get_logger()
    start = time.perf_counter()

    for b_num, b in batches:
        systems = []
        systems_results_paths = []
        for system_to_calculate in b:
            path_system_to_calculate = os.path.join(cif_directory, system_to_calculate)
            path_system_to_calculate_results = os.path.join(results_directory, '{}__running'.format(system_to_calculate))
            os.mkdir(path_system_to_calculate_results)

            with open(path_system_to_calculate) as out:
                smi = out.readlines()[0]
            
            systems.append(smi), systems_results_paths.append(path_system_to_calculate_results)



        eval_logger.info(f'Preparing substrates batch {b_num}')
        substrates = multi_prepare_substrate(systems, calc_kwargs=CALC_PARAMS, multi_process=-1)
        ### Implement multiprocessing.Pool error handling here. I presume errors wouldn't arrise here, but you never know

        intermediates = []
        for j, (sub, res_path) in enumerate(zip(substrates, systems_results_paths)):
            os.chdir(res_path)

            eval_logger.info('Building and relaxing configurations')
            oh_configs, o_configs, ooh_configs = build_and_relax_configurations(sub, sub.info['equivalent_atoms'], optlevel='loose', multi_process=-1, additional_conformers=False)
            oh_stable, o_stable, ooh_stable = global_min_configurations(oh_configs, o_configs, ooh_configs)
            intermediates.append([oh_stable, o_stable, ooh_stable])
        


        for j, system_to_calculate in enumerate(b):
            print(f'Iter: {j} / {total_num_systems}')
            print(system_to_calculate)
            start_time = time.perf_counter()
            path_system_to_calculate = os.path.join(cif_directory, system_to_calculate)
            path_system_to_calculate_results = os.path.join(results_directory, '{}__running'.format(system_to_calculate))

            os.mkdir(path_system_to_calculate_results)
            os.chdir(path_system_to_calculate_results)

            # run_command = ['python', SCRIPT, path_system_to_calculate]
            # success = run_evaluation_subprocess(run_command)
            success = run_evaluation()

            os.system('echo {} >> results.txt'.format(time.perf_counter() - start_time))
            os.chdir('..')

            if success:  # Handling success completion
                state_new = "completed"
            else:  # Handling error state
                state_new = "fizzled"

            os.system("mv {0}__running {0}__{1}".format(system_to_calculate, state_new)) #rename file according to new state
            print('#################')