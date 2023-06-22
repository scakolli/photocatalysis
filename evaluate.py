import numpy as np
from copy import deepcopy
import time

from photocatalysis.adsorption.tools import prepare_substrate
from photocatalysis.adsorption.relaxing import build_and_relax_configurations
from photocatalysis.thermodynamics.tools import multi_run, free_energies
from photocatalysis.thermodynamics.constants import SHE_VACUUM_POTENTIAL, WATER_OXIDATION_POTENTIAL
from photocatalysis.thermodynamics.helpers import get_logger


def evaluate_substrate(smile_string, calculator_params):
    eval_logger = get_logger()
    eval_logger.info('Preparing substrate')
    start = time.perf_counter()

    ### Prepare substrate
    # Generate FF optimized confs, optimize lowest energy one at the tight binding level, calculate ZPE-TS and IP/EA
    substrate = prepare_substrate(smile_string, calculator_params, multi_process_conf=2, multi_process_sp=4)
    sites = substrate.info['equivalent_atoms']

    ### Relax and filter
    # Crude relaxation is sufficient
    eval_logger.info('Building and relaxing configurations')
    oh_configs, o_configs, ooh_configs = build_and_relax_configurations(substrate, sites, optlevel='loose', multi_process=6, additional_conformers=False)

    ### Rate determining free energy and other quantities
    eval_logger.info('Calculating thermochemical properties')
    asites, rds, rdg, essi = calculate_thermochemistry(substrate, oh_configs, o_configs, ooh_configs)
    driving_potential = substrate.info['ip'] / 1. - SHE_VACUUM_POTENTIAL

    if driving_potential > rdg:
        print("Substrate likely suitable for water oxidation")
        print(f"Driving Potential (V) {driving_potential} > Rate Determining Potential (V) {rdg / 1.0}")

    eval_logger.info(f'Evaluation Took {time.perf_counter() - start}s')
    print('#######################')
    return driving_potential, rdg, essi, asites, rds


def calculate_thermochemistry(substrate, oh_configs, o_configs, ooh_configs):
    calculator_params = deepcopy(substrate.info['calc_params'])

    ### Determine most stable set of intermediates, and completely optimize them (and perform ZPE/TS calc)
    Eoh = np.array([config.info['energy'] for config in oh_configs])
    Eo = np.array([config.info['energy'] for config in o_configs])
    Eooh = np.array([config.info['energy'] for config in ooh_configs])

    min_energy_configs = [oh_configs[Eoh.argmin()], o_configs[Eo.argmin()], ooh_configs[Eooh.argmin()]]
    oh_stable, o_stable, ooh_stable = multi_run(min_energy_configs, runtype='ohess vtight', calc_kwargs=calculator_params, multi_process=3)

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

    return active_sites, rate_det_step, rate_det_energy, ESSI
