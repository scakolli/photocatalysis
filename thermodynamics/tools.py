import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import subprocess

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

def single_point(molecule, method="GFN2-xTB", accuracy=0.2, electronic_temperature=298.15, solvent=None,
                 relaxation=False, trajectory=None, fmax=0.05):
    # Attach Caclulator to ASE molecule
    mol = molecule.copy()
    mol.calc = XTB(method=method, accuracy=accuracy, electronic_temperature=electronic_temperature, solvent=str(solvent))

    # Locally optimize geometry
    if relaxation:
        mol.info['opt_file'] = trajectory
        logger = get_logger()
        logger.info('Optimizing Geometry')
        optimizer = LBFGS(mol, trajectory=trajectory, logfile=None)
        optimizer.run(fmax=fmax)

    # Calculate Energy and delete calc so you can pickle molecule w/o errors
    mol.info['energy'] = mol.get_potential_energy()
    del mol.calc

    return mol

def single_point_worker(jobs):
    molecule, calc_param_kwargs, opt_param_kwargs = jobs
    return single_point(molecule, **calc_param_kwargs, **opt_param_kwargs)

def ipea(molecule, calc_params, n_cores=4):

    molecule.write('scratch.xyz')

    cmd = 'xtb scratch.xyz --vipea --acc {} --etemp {} --parallel {} --ceasefiles'.format(
        calc_params['accuracy'],
        calc_params['electronic_temperature'],
        n_cores)

    out = subprocess.run(cmd.split(), capture_output=True)

    if out.returncode == 0:
        string_out = out.stdout.decode('UTF-8')
        ip, ea, homo, lumo = parse_ipea(string_out)
        molecule.info['ip'] = ip
        molecule.info['ea'] = ea
        molecule.info['homo'] = homo
        molecule.info['lumo'] = lumo

        return ip, ea, homo, lumo
    else:
        print("Error, Check Return Code")
        return out

def parse_ipea(string):
    # Parse IP/EA aswell as HOMO/LUMO for comparison
    homo_parsed, lumo_parsed = False, False
    for line in string.splitlines():
        if 'delta SCC IP (eV)' in line:
            # ionization potential in eV
            ip = float(line.split()[4])
        if 'delta SCC EA (eV)' in line:
            # electron affinity in eV
            ea = float(line.split()[4])
        if '(HOMO)' in line and not homo_parsed:
            # KS Homo in eV
            homo = float(line.split()[3])
            homo_parsed = True
        if '(LUMO)' in line and not lumo_parsed:
            # KS Lumo in eV
            lumo = float(line.split()[2])
            lumo_parsed = True

    return ip, ea, homo, lumo

def zero_point_energy(molecule, calculator_params, n_cores=4):
    ## Try std_output: except error
    ## Introduce solvent

    molecule.write('scratch.xyz')

    cmd = 'xtb --gfn {} scratch.xyz --hess --acc {} --etemp {} --parallel {} --ceasefiles --silent --strict'.format(
        calculator_params['method'][3],
        calculator_params['accuracy'],
        calculator_params['electronic_temperature'],
        n_cores)

    out = subprocess.run(cmd.split(), capture_output=True)

    if out.returncode == 0:
        ## Hessian calculation ran successfully, exit
        string_out = out.stdout.decode('UTF-8')
        zpe = parse_zpe(string_out)
        molecule.info['zpe'] = zpe

        return zpe
    else:
        return out
        print("Error: Incompletely Optimized Geometry")

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
