import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import subprocess
from copy import deepcopy
import shutil
from itertools import dropwhile, takewhile

from xtb.ase.calculator import XTB
from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MUTED
from xtb.interface import Calculator, Param
from xtb.utils import get_method as get_xtb_method

import ase
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from ase.units import Hartree, Bohr
from osc_discovery.photocatalysis.thermodynamics.constants import dG1_CORR, dG2_CORR, dG3_CORR, dG4_CORR

def get_logger():
    logger_ = logging.getLogger()
    logger_.setLevel(logging.INFO)
    if not logger_.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s: %(message)s'))
        logger_.addHandler(console_handler)
    return logger_

def single_point_worker(jobs):
    molecule, calc_param_kwargs, opt_param_kwargs = jobs
    return single_point(molecule, **calc_param_kwargs, **opt_param_kwargs)

def single(molecule, runtype='sp', keep_folder=False, **calculator_kwargs):
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

    assert "OMP_NUM_THREADS" in os.environ, "'OMP_NUM_THREADS' env var not set, unparallelized calc, very slow"
    mol = deepcopy(molecule)

    # Create folder
    num_run_folders_cwd = len([folder for folder in os.listdir() if 'run_' in folder])
    fname = f'run_{num_run_folders_cwd}'
    os.mkdir(fname)
    os.chdir(fname)

    # Build command
    mol.write('scratch.xyz')
    cmd = f'xtb scratch.xyz --{runtype} --strict'
    for key, value in calculator_kwargs.items():
        cmd += f" --{key} {value}"

    # Execute command
    process_output = subprocess.run((cmd), shell=True, capture_output=True)
    stdoutput = process_output.stdout.decode('UTF-8')

    if process_output.returncode != 0:
        # Abnormal termination of xtb, errors are encapsulated by '###'
        os.chdir('..')
        error = list(dropwhile(lambda line: "###" not in line, stdoutput.splitlines()))
        raise RuntimeError('Abnormal termination of xtb \n'+'\n'.join(error))

    # Update molecule geometry, parse output for properties and attach info to molecule, clean up folders
    if 'opt' in runtype:
        mol = ase.io.read('xtbopt.xyz')
        del mol.info
        mol.info = deepcopy(molecule.info)

    out_dict = parse_stdoutput(stdoutput, runtype)
    mol.info.update(out_dict)

    os.chdir('..')
    if keep_folder:
        mol.info['fname'] = os.getcwd()
    else:
        shutil.rmtree(fname)

    return mol

def parse_stdoutput(xtb_output, runtype):
    # Standard output from xtb call is parsed according to runtype
    d = dict()
    if runtype == 'sp' or 'opt' in runtype:
        d['energy'], d['ehomo'], d['elumo'] = parse_energies(xtb_output)
    elif runtype == 'hess':
        d['zpe'] = parse_zpe(xtb_output)
        # Thermo parsing
    elif runtype == 'vipea':
        d['ip'], d['ea'], d['ehomo'], d['elumo'] = parse_ipea(xtb_output)

    return d

def parse_energies(string):
    for line in string.splitlines():
        if '(HOMO)' in line:
            # KS Homo in eV
            homo = float(line.split()[-2])
        if '(LUMO)' in line:
            # KS Lumo in eV
            lumo = float(line.split()[-2])
        if 'TOTAL' in line:
            # Total Energy in eV
            e = float(line.split()[-3]) * Hartree

    return e, homo, lumo

def parse_ipea(string):
    # Parse IP/EA aswell as HOMO/LUMO for comparison
    homo_parsed, lumo_parsed = False, False
    for line in string.splitlines():
        if 'delta SCC IP (eV)' in line:
            # ionization potential in eV
            ip = float(line.split()[-1])
        if 'delta SCC EA (eV)' in line:
            # electron affinity in eV
            ea = float(line.split()[-1])
        if '(HOMO)' in line and not homo_parsed:
            # KS Homo in eV
            homo = float(line.split()[-2])
            homo_parsed = True
        if '(LUMO)' in line and not lumo_parsed:
            # KS Lumo in eV
            lumo = float(line.split()[-2])
            lumo_parsed = True

    return ip, ea, homo, lumo

def parse_zpe(string):
    for line in string.splitlines():
        if 'zero point energy' in line:
            zpe_ = float(line.split()[4]) * Hartree # in eV
            return zpe_



def calculate_free_energies(s, OH, O, OOH):
    ### Calculate reaction free energies of *, *OH, *O, *OOH species
    if 'zpe' not in s.info:
        ZPEs = zero_point_energy(s, s.info['calc_params'])
        ZPEOH = zero_point_energy(OH, s.info['calc_params'])
        ZPEO = zero_point_energy(O, s.info['calc_params'])
        ZPEOOH = zero_point_energy(OOH, s.info['calc_params'])
    else:
        ZPEs = s.info['zpe']
        ZPEOH = OH.info['zpe']
        ZPEO = O.info['zpe']
        ZPEOOH = OOH.info['zpe']

    Es, EOH, EO, EOOH = s.info['energy'], OH.info['energy'], O.info['energy'], OOH.info['energy']

    dG1 = (EOH + ZPEOH) - (Es + ZPEs) + dG1_CORR
    dG2 = (EO + ZPEO) - (EOH + ZPEOH) + dG2_CORR
    dG3 = (EOOH + ZPEOOH) - (EO + ZPEO) + dG3_CORR
    dG4 = (Es + ZPEs) - (EOOH + ZPEOOH) + dG4_CORR

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
