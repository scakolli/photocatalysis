from copy import deepcopy
import time
import os
import traceback

from photocatalysis.adsorption.tools import prepare_substrate, multi_prepare_substrate
from photocatalysis.adsorption.relaxing import build_and_relax_configurations
from photocatalysis.thermodynamics.thermodynamics import get_thermodynamics, multi_get_thermodynamics

from photocatalysis.thermodynamics.helpers import get_logger

def evaluate_substrate(smile_string, calculator_params, scratch_dir=None):
    if scratch_dir is not None:
        base = os.getcwd()
        os.chdir(scratch_dir)

    eval_logger = get_logger()
    eval_logger.info('Preparing substrate')
    start = time.perf_counter()

    ### Prepare substrate
    # Generate FF optimized confs, optimize lowest energy one at the tight binding level, calculate ZPE-TS and IP/EA
    substrate = prepare_substrate(smile_string, calculator_params, multi_process_conf=4, multi_process_sp=4)
    sites = substrate.info['equivalent_atoms']

    ### Relax and filter
    # Crude relaxation is sufficient
    eval_logger.info('Building and relaxing configurations')
    oh_configs, o_configs, ooh_configs = build_and_relax_configurations(substrate, sites, optlevel='loose', multi_process=-1)

    ### Rate determining free energy and other quantities
    eval_logger.info('Calculating thermochemical properties')
    driving_potential, rdg, asites, rds, essi = get_thermodynamics(substrate, oh_configs, o_configs, ooh_configs, multi_processing=3)

    if driving_potential > rdg:
        print("Substrate likely suitable for water oxidation")
        print(f"Driving Potential (V) {driving_potential} > Rate Determining Potential (V) {rdg / 1.0}")

    eval_logger.info(f'Evaluation Took {time.perf_counter() - start}s')

    if scratch_dir is not None:
        os.chdir(base)

    return driving_potential, rdg, asites, rds, essi

def evaluate_substrate_in_batches(smile_strings, calculator_params, scratch_dir=None, batch_number=None):
    ### smile_string which is attched to each substrate is used to identify molecules during each step
    ### If not smile string, then you could use the folder created for the run as an identifier
    if scratch_dir is not None:
        base = os.getcwd()
        os.chdir(scratch_dir)

    eval_logger = get_logger()

    ##### Multiprocess and catch substrate prep errors (implicit error handling done with multiprocess.imap)
    print('############################')
    eval_logger.info(f'PREPARING SUBSTRATES, BATCH {batch_number}')

    preped_subs, prep_errors = multi_prepare_substrate(smile_strings, calc_kwargs=calculator_params)
    eval_logger.info(f'FIZZLED: {len(prep_errors)}')

    ##### Relax configs in for loop (explicit error handling with try/except block)
    print('############################')
    eval_logger.info(f'BUILDING AND RELAXING CONFIGURATIONS, BATCH {batch_number}')

    relaxed_systems = []
    relax_errors = []
    for j, (smi, sub) in enumerate(preped_subs):
        print(f'\n ITER {j}, {smi}')
        try:
            oh, o, ooh = build_and_relax_configurations(sub, sub.info['equivalent_atoms'])
            relaxed_systems.append([sub, oh, o, ooh])
        except Exception as e:
            # process errors here
            print(f'ERROR IN {smi}')
            relax_errors.append((smi, traceback.format_exc()))

    eval_logger.info(f'FIZZLED: {len(relax_errors)}')
    ##### Get thermodynamic variables of interest (implicit error handling)
    print('############################')
    eval_logger.info(f'THERMODYNAMIC ASSESMENT, BATCH {batch_number}')
    
    properties, prop_errors = multi_get_thermodynamics(relaxed_systems)
    eval_logger.info(f'FIZZLED: {len(prop_errors)}') 

    print('############################')
    total_errors = prep_errors + relax_errors + prop_errors
    eval_logger.info(f'SUCCESSFULLY RAN: {len(smile_strings) - len(total_errors)} / {len(smile_strings)}')
    eval_logger.info('IP > dGMAX COUNT: {}'.format(sum([prop[0] > prop[1] for _, prop in properties])))

    if scratch_dir is not None:
        os.chdir(base)

    return properties, total_errors
