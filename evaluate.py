import numpy as np
from copy import deepcopy
import time
import os
import multiprocessing
from itertools import repeat

from photocatalysis.adsorption.tools import prepare_substrate
from photocatalysis.adsorption.relaxing import build_and_relax_configurations
from photocatalysis.thermodynamics.tools import single_run, multi_run, get_multi_process_cores, free_energies
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
    oh_configs, o_configs, ooh_configs = build_and_relax_configurations(substrate, sites, optlevel='loose', multi_process=-1, additional_conformers=False)

    ### Rate determining free energy and other quantities
    eval_logger.info('Calculating thermochemical properties')
    rdg, asites, rds, essi = calculate_thermochemistry(substrate, oh_configs, o_configs, ooh_configs, multi_process=-1)
    driving_potential = substrate.info['ip'] / 1. - SHE_VACUUM_POTENTIAL

    if driving_potential > rdg:
        print("Substrate likely suitable for water oxidation")
        print(f"Driving Potential (V) {driving_potential} > Rate Determining Potential (V) {rdg / 1.0}")

    eval_logger.info(f'Evaluation Took {time.perf_counter() - start}s')

    if scratch_dir is not None:
        os.chdir(base)

    return driving_potential, rdg, asites, rds, essi

def global_min_configurations(oh_configs, o_configs, ooh_configs):
    Eoh = np.array([config.info['energy'] for config in oh_configs])
    Eo = np.array([config.info['energy'] for config in o_configs])
    Eooh = np.array([config.info['energy'] for config in ooh_configs])

    min_energy_configs = [oh_configs[Eoh.argmin()], o_configs[Eo.argmin()], ooh_configs[Eooh.argmin()]]

    return min_energy_configs

def calculate_thermochemistry(substrate, oh_configs, o_configs, ooh_configs, multi_process=1, job_number=0):
    assert (multi_process == 1) or (multi_process == -1), 'Either single core or 3 core evaluation supported'

    #  Determine most stable set of intermediates
    min_energy_configs = global_min_configurations(oh_configs, o_configs, ooh_configs)

    # Completely optimize them (and perform ZPE/TS calc)
    calculator_params = deepcopy(substrate.info['calc_params'])
    if multi_process == -1:
        # Only 3 intermediates are analyzed here...
        # use additional cores for each intermediate (4 gives best walltimes)
        calculator_params.update({'parallel':4})
        oh_stable, o_stable, ooh_stable = multi_run(min_energy_configs, runtype='ohess vtight', calc_kwargs=calculator_params, multi_process=multi_process)
    else:
        oh_stable = single_run(min_energy_configs[0], runtype='ohess vtight', job_number=job_number, **calculator_params)
        o_stable = single_run(min_energy_configs[1], runtype='ohess vtight', job_number=job_number, **calculator_params)
        ooh_stable = single_run(min_energy_configs[2], runtype='ohess vtight', job_number=job_number, **calculator_params)

    ### Free energies of intermediates
    gs = substrate.info['energy'] +  substrate.info['zpe'] - substrate.info['entropy']
    goh = oh_stable.info['energy'] + oh_stable.info['zpe'] - oh_stable.info['entropy']
    go = o_stable.info['energy'] + o_stable.info['zpe'] - o_stable.info['entropy']
    gooh = ooh_stable.info['energy'] + ooh_stable.info['zpe'] - ooh_stable.info['entropy']

    ### Reaction free energy changes and thermodynamic quantites of interest
    G = free_energies(gs, goh, go, gooh)

    rate_det_step = G.argmax()
    rate_det_energy = G.max() # basically overpotential
    ESSI = G[G > 1.23].sum() / G[G > 1.23].size # Electrochemical-Step-Symmetry-Index (similar to Mean Abs Diff)

    active_sites = [oh_stable.info['active_site'], o_stable.info['active_site'], ooh_stable.info['active_site']]

    return rate_det_energy, active_sites, rate_det_step, ESSI

def calculate_thermochemistry_worker(job):
    job_num, job_input = job
    substrate, oh_configs, o_configs, ooh_configs = job_input
    return calculate_thermochemistry(substrate, oh_configs, o_configs, ooh_configs, job_number=job_num)

def multi_calculate_thermochemistry(systems, calc_kwargs=None, multi_process=-1):
    # var : systems, comprised of [[substrate, oh_configs, o_configs, ooh_configs], ... ]
    # Generate (job_number, (single_run_parameters)) jobs to send to 
    multi_process = get_multi_process_cores(len(systems), multi_process) # multi_process=-1 returns returns max efficient cores, else return multi_process
    jobs = list(enumerate(zip(systems, repeat(calc_kwargs))))

    job_logger = get_logger()
    print('Devoted Cores:', multi_process)
    job_logger.info(f'substrate preparation jobs to do: {len(systems)}')

    start = time.perf_counter()
    if multi_process == 1:
        # Serial Relaxation
        completed_molecule_list = list(map(calculate_thermochemistry_worker, jobs))
    else:
        # Parallel Relaxation
        with multiprocessing.Pool(multi_process) as pool:
            completed_molecule_list = pool.map(calculate_thermochemistry_worker, jobs)

    job_logger.info(f'finished jobs. Took {time.perf_counter() - start}s')

    return completed_molecule_list
