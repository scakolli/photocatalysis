import numpy as np
from copy import deepcopy
import time
import os
from itertools import repeat

from photocatalysis.adsorption.tools import prepare_substrate, multi_prepare_substrate
from photocatalysis.adsorption.relaxing import build_and_relax_configurations
from photocatalysis.thermodynamics.thermodynamics import get_thermodynamics, multi_get_thermodynamics

from photocatalysis.thermodynamics.constants import SHE_VACUUM_POTENTIAL
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
    rdg, asites, rds, essi = get_thermodynamics(substrate, oh_configs, o_configs, ooh_configs, multi_processing=3)
    driving_potential = substrate.info['ip'] / 1. - SHE_VACUUM_POTENTIAL

    if driving_potential > rdg:
        print("Substrate likely suitable for water oxidation")
        print(f"Driving Potential (V) {driving_potential} > Rate Determining Potential (V) {rdg / 1.0}")

    eval_logger.info(f'Evaluation Took {time.perf_counter() - start}s')

    if scratch_dir is not None:
        os.chdir(base)

    return driving_potential, rdg, asites, rds, essi

def evaluate_substrate_in_batches(smile_strings, calculator_params, results_folders):
    eval_logger = get_logger()

    ##### Multiprocess and catch substrate prep errors (implicit error handling done with multiprocess.imap)
    print('############################')
    eval_logger.info(f'PREPARING SUBSTRATES BATCH')

    subs, errors = multi_prepare_substrate(smile_strings, calc_kwargs=calculator_params)
    # process errors here

    ##### Relax configs in for loop (explicit error handling with try/except block)
    print('############################')
    eval_logger.info('BUILDING AND RELAXING CONFIGURATIONS BATCH')

    systems = []
    for j, sub in enumerate(subs):
        print(f'Iter {j}')
        try:
            oh, o, ooh = build_and_relax_configurations(sub, sub.info['equivalent_atoms'])
            systems.append([sub, oh, o, ooh])
        except Exception as e:
            # process errors here
            pass
    
    ##### Get thermodynamic variables of interest (implicit error handling)
    print('############################')
    eval_logger.info('THERMODYNAMIC ASSESMENT BATCH')

    props, errors_props = multi_get_thermodynamics(systems)
    ips = [s.info['ip'] / 1. - SHE_VACUUM_POTENTIAL for s in subs]

    # Process last bit of errors here

    # for smi, ip, prop in zip(systems, driving_potentials, properties):
    #         rdg, asites, rds, essi = prop
    #         os.system('echo "{} {} {} {} {} {}" >> {}/results_calculations_test.txt'.format(smi, ip, rdg, asites, rds, essi, results_directory))

    #return ips, props
