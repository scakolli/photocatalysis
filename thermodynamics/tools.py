import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import subprocess
from copy import deepcopy
import shutil
from itertools import dropwhile, takewhile, repeat
import time
import multiprocessing

import ase
from osc_discovery.photocatalysis.thermodynamics.constants import dG1_CORR, dG2_CORR, dG3_CORR, dG4_CORR
from osc_discovery.photocatalysis.thermodynamics.helpers import parse_stdoutput, parse_charges
from osc_discovery.photocatalysis.thermodynamics.helpers import xtboptlog_to_ase_trajectory

def get_logger():
    logger_ = logging.getLogger()
    logger_.setLevel(logging.INFO)
    if not logger_.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s: %(message)s'))
        logger_.addHandler(console_handler)
    return logger_

def single_run(molecule, runtype='sp', keep_folder=False, job_number=0, **calculator_kwargs):
    """
    Execute XTB calculations on molecule
    (see also man page: https://github.com/grimme-lab/xtb/blob/main/man/xtb.1.adoc)

    ### Runtypes
    # sp : single-point scc calc
    # opt [LEVEL]: geometry optimization, e.g. 'opt vtight'
    # hess : vibrational and thermodynamic analysis
    # vipea : vertical ionization potential and electron affinity (This needs the .param_ipea.xtb parameters and a GFN1 Hamiltonian)
    # vfukui : Fukui indices

    ### Calculator Options
    # (calculator_kwarg : default_value, description)
    # gfn : 2, parameterization
    # chrg : 0, charge
    # uhf : 0, unpaired electrons
    # acc : 1.0, accuracy
    # etemp : 300, electronic temp

    # gbsa : None, implicit solvation with gbsa
    # alpb : None, implicit solvation with alpb
    # parallel: None, number of cores to devote
    # ceasefiles: , stop file printout (not all files are always halted)
    """
    error_logger = get_logger()
    assert "OMP_NUM_THREADS" in os.environ, "'OMP_NUM_THREADS' env var not set, unparallelized calc, very slow"
    mol = deepcopy(molecule)

    # Create folder
    fname = f'run_{job_number}'
    os.mkdir(fname)
    os.chdir(fname)

    # Build command
    mol.write('scratch.xyz')
    cmd = f'xtb scratch.xyz --{runtype}'
    for key, value in calculator_kwargs.items():
        cmd += f" --{key} {value}"

    # Execute command
    process_output = subprocess.run((cmd), shell=True, capture_output=True)
    stdoutput = process_output.stdout.decode('UTF-8')
    ################# Error Handling #################
    if process_output.returncode != 0:
        error_logger.error(f'Runtime Errors Encountered in job {job_number}, attempting to solve')
        # Abnormal termination of xtb, errors are encapsulated by '###'
        error = list(dropwhile(lambda line: "###" not in line, stdoutput.splitlines()))

        if '-1- scf: Self consistent charge iterator did not converge' in error:
            # Calculation/optimization failed to converge because of lack scf convergence
            # Raise fermi-smearing temp (1000 K) and restart original calculation (298.15K)
            # from 'xtbrestart' created during the hot run
            try:
                # Basically, try: 'xtb scratch.xyz --etemp 1000 && xtb scratch.xyz --restart'
                cmd_hot_restart = cmd.replace('--etemp {}'.format(calculator_kwargs['etemp']), f'--etemp 1000')
                cmd_hot_restart += ' && ' + cmd + ' --restart'
            except KeyError:
                # etemp not in calc params... introduce it
                cmd_hot_restart = cmd + ' --etemp 1000' + ' && ' + cmd + ' --restart'

            process_output = subprocess.run((cmd_hot_restart), shell=True, capture_output=True)
            stdoutput = process_output.stdout.decode('UTF-8')

            if process_output.returncode != 0:
                if 'opt' in runtype:
                    # Last ditch effort at getting convergence
                    # If we are optimizing, we can use xtbopt.xyz at 1000 K as a start geometry, and optimize
                    # at 300 K from there
                    cmd_last = cmd.replace('scratch.xyz', 'xtbopt.xyz')
                    process_output = subprocess.run((cmd_last), shell=True, capture_output=True)
                    stdoutput = process_output.stdout.decode('UTF-8')

                    if process_output.returncode != 0:
                        os.chdir('..')
                        raise RuntimeError('optimizing from high temp config failed to help scf convergence')

                else:
                    os.chdir('..')
                    raise RuntimeError('restart with etemp increase failed to help scf convergence')

        else:
            os.chdir('..')
            raise RuntimeError('Abnormal termination of xtb \n'+'\n'.join(error))
    ################# Error Handling Finished #################

    # Update molecule geometry, parse output for properties and attach info to molecule, clean up folders
    if 'opt' in runtype:
        mol = ase.io.read('xtbopt.xyz')
        del mol.info
        mol.info = deepcopy(molecule.info)

    out_dict = parse_stdoutput(stdoutput, runtype)
    if 'pop' in calculator_kwargs:
        out_dict.update({'qs' : parse_charges()}) # Request a Mulliken population analysis charges
    mol.info.update(out_dict)

    if keep_folder:
        mol.info['fname'] = os.path.join(os.getcwd(), fname)
        xtboptlog_to_ase_trajectory('xtbopt.log', f'opt{job_number}.traj')
        os.chdir('..')
    else:
        os.chdir('..')
        shutil.rmtree(fname)

    return mol

def single_run_worker(job):
    # job: tuple( job_number, (molecule, runtype, keep_folder boolean, calc_kwargs_dictionary) )
    # Unpack job and return single_run() worker that can be used in multiprocessing code
    job_num, job_input = job
    mol, runt, keepf, calc_kwargs = job_input
    return single_run(mol, runtype=runt, keep_folder=keepf, job_number=job_num, **calc_kwargs)

def multi_run(molecule_list, runtype='opt', keep_folders=False, calc_kwargs=None, multi_process=1):
    # Generate (job_number, (single_run_parameters)) jobs to send to worker
    jobs = list(enumerate(zip(molecule_list, repeat(runtype), repeat(keep_folders), repeat(calc_kwargs))))

    job_logger = get_logger()
    job_logger.info(f'Jobs to do: {len(molecule_list)}')
    start = time.perf_counter()
    if multi_process == 1:
        # Serial Relaxation
        completed_molecule_list = list(map(single_run_worker, jobs))
    else:
        # Parallel Relaxation
        with multiprocessing.Pool(multi_process) as pool:
            completed_molecule_list = pool.map(single_run_worker, jobs)

    job_logger.info(f'Finished jobs. Took {time.perf_counter() - start}s')

    return completed_molecule_list

def calculate_free_energies(s, OH, O, OOH):
    ### Calculate reaction free energies of *, *OH, *O, *OOH species
    assert 'zpe' in OH.info, "No frequency analysis done yet"
    Es, EOH, EO, EOOH = s.info['energy'], OH.info['energy'], O.info['energy'], OOH.info['energy']
    ZPEs, ZPEOH, ZPEO, ZPEOOH = s.info['zpe'], OH.info['zpe'], O.info['zpe'], OOH.info['zpe']
    TSs, TSOH, TSO, TSOOH = s.info['entropy'], OH.info['entropy'], O.info['entropy'], OOH.info['entropy']

    dG1 = (EOH + ZPEOH - TSOH) - (Es + ZPEs - TSs) + dG1_CORR
    dG2 = (EO + ZPEO - TSO) - (EOH + ZPEOH - TSOH) + dG2_CORR
    dG3 = (EOOH + ZPEOOH - TSOOH) - (EO + ZPEO - TSO) + dG3_CORR
    dG4 = (Es + ZPEs - TS) - (EOOH + ZPEOOH - TSOOH) + dG4_CORR

    Gs = np.array((dG1, dG2, dG3, dG4))

    return Gs

def free_energies(Es, EOH, EO, EOOH, ZPE_TS=None):
    ### Calculate reaction free energies of *, *OH, *O, *OOH species

    # if ZPE_TS is None:


    dG1 = (EOH + ZPEOH - TSOH) - (Es + ZPEs - TSs) + dG1_CORR
    dG2 = (EO + ZPEO - TSO) - (EOH + ZPEOH - TSOH) + dG2_CORR
    dG3 = (EOOH + ZPEOOH - TSOOH) - (EO + ZPEO - TSO) + dG3_CORR
    dG4 = (Es + ZPEs - TS) - (EOOH + ZPEOOH - TSOOH) + dG4_CORR

    Gs = np.array((dG1, dG2, dG3, dG4))

    return Gs

def free_energy_diagram(Gs):

    x = [0, 1, 2, 3, 4]
    y = [0]+Gs.cumsum().tolist() # No bias
    y_123 = [0] + (Gs - 1.23).cumsum().tolist() # Equilibrium bias 1.23V
    y_downhill = [0] + (Gs - Gs.max()).cumsum().tolist() # Bias when all steps are down hill

    plt.step(x, y, 'k', label='0V')
    plt.step(x, y_123, '--k', label='1.23V')
    plt.step(x, y_downhill, 'b', label='{}V'.format(round(Gs.max(), 2)))

    plt.xlabel('Intermediates')
    plt.ylabel('Free Energy (eV)')
    plt.legend()
