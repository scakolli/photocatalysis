import time
import multiprocessing
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from photocatalysis.thermodynamics.tools import single_run, multi_run
from photocatalysis.thermodynamics.constants import dG1_REST, dG2_REST, dG3_REST, dG4_REST
from photocatalysis.thermodynamics.helpers import get_multi_process_cores, explicitly_broadcast_to, get_logger
from photocatalysis.adsorption.helpers import multiprocessing_run_and_catch

def free_energies(Gs, GOH, GO, GOOH):
    # Given the free energies of each intermediate (could also just be the energies, if you want a non-zpe/ts approx.)
    # determine the free energy changes of each step in the proposed reaction mechanism
    dG1 = GOH - Gs + dG1_REST
    dG2 = GO - GOH + dG2_REST
    dG3 = GOOH - GO + dG3_REST
    dG4 = Gs - GOOH + dG4_REST

    return np.array([dG1, dG2, dG3, dG4])

def free_energies_multidim(Gs, GOH, GO, GOOH, explicitly_broadcast=True):
    # Vectorized free energy expressions with numpy broadcasting.
    # (Gs) : scalar, (GOH, GO, GOOH) : arbitrary length numpy arrays corresponding to each configuration
    # Access free energy expression of configs (3,5,6) / (OH_index, O_index, OOH_index) with for example
    # G[3,5,6] -> [1.24,1.22,1.24,1.22]

    g1 = GOH[:, None, None] - Gs + dG1_REST
    g2 = GO[None, :, None] - GOH[:, None, None] + dG2_REST
    g3 = GOOH[None, None, :] - GO[None, :, None] + dG3_REST
    g4 = Gs - GOOH[None, None, :] + dG4_REST

    if explicitly_broadcast:
        tot_shape = len(GOH), len(GO), len(GOOH)
        g1b, g2b, g3b, g4b = explicitly_broadcast_to(tot_shape, g1, g2, g3, g4)
        G = np.moveaxis(np.array((g1b, g2b, g3b, g4b)), 0, 3)
        return G
    else:
        return g1, g2, g3, g4

def free_energy_diagram(Gs_array):
    # quick and dirty plotting of free energies

    x = [0, 1, 2, 3, 4] # steps
    y = [0]+Gs_array.cumsum().tolist() # No bias
    y_123 = [0] + (Gs_array - 1.23).cumsum().tolist() # Equilibrium bias 1.23V
    y_downhill = [0] + (Gs_array - Gs_array.max()).cumsum().tolist() # Bias when all steps are down hill

    plt.step(x, y, 'k', label='0V')
    plt.step(x, y_123, '--k', label='1.23V')
    plt.step(x, y_downhill, 'b', label='{}V'.format(round(Gs_array.max(), 2)))

    plt.xlabel('Intermediates')
    plt.ylabel('Free Energy (eV)')
    plt.legend()

def global_min_configurations(oh_configs, o_configs, ooh_configs):
    Eoh = np.array([config.info['energy'] for config in oh_configs])
    Eo = np.array([config.info['energy'] for config in o_configs])
    Eooh = np.array([config.info['energy'] for config in ooh_configs])

    min_energy_configs = [oh_configs[Eoh.argmin()], o_configs[Eo.argmin()], ooh_configs[Eooh.argmin()]]

    return min_energy_configs

def get_thermodynamics(substrate, oh_configs, o_configs, ooh_configs, multi_processing=None, job_number=0):
    #  Determine most stable set of intermediates
    min_energy_configs = global_min_configurations(oh_configs, o_configs, ooh_configs)

    # Completely optimize them (and perform ZPE/TS calc)
    calculator_params = deepcopy(substrate.info['calc_params'])
    if multi_processing is None:
        # calculator_params.update({'parallel':4})
        # stable = [single_run(config, runtype='opt vtight', job_number=job_number, **calculator_params) for config in min_energy_configs]
        # oh_stable, o_stable, ooh_stable = [single_run(config, runtype='hess', job_number=job_number, **calculator_params) for config in stable]
        oh_stable, o_stable, ooh_stable = [single_run(config, runtype='ohess vtight', job_number=job_number, **calculator_params) 
                                           for config in min_energy_configs]
    else:
        # Only 3 intermediates are analyzed here...
        # use additional cores for each intermediate (4 gives best walltimes)
        calculator_params.update({'parallel':4})
        oh_stable, o_stable, ooh_stable = multi_run(min_energy_configs, runtype='ohess vtight', calc_kwargs=calculator_params, multi_process=multi_processing)

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

def get_thermodynamics_worker(job):
    job_num, job_input = job
    substrate, oh_configs, o_configs, ooh_configs = job_input
    return get_thermodynamics(substrate, oh_configs, o_configs, ooh_configs, job_number=job_num)

def multi_get_thermodynamics(systems, multi_process=-1):
    # var : systems, comprised of [[substrate, oh_configs, o_configs, ooh_configs], ... ]
    # Generate (job_number, (single_run_parameters)) jobs to send to 
    multi_process = get_multi_process_cores(len(systems), multi_process) # multi_process=-1 returns returns max efficient cores, else return multi_process
    jobs = list(enumerate(systems))

    job_logger = get_logger()
    job_logger.info(f'thermodynamic (ZPE-TS) jobs to do: {len(systems)}')

    start = time.perf_counter()

    # Parallel Relaxation
    with multiprocessing.Pool(multi_process) as pool:
        systems_iterator = pool.imap(get_thermodynamics_worker, jobs)
        systems_properties, systems_errors = multiprocessing_run_and_catch(systems_iterator)

    job_logger.info(f'finished jobs. Took {time.perf_counter() - start}s')

    return systems_properties, systems_errors
