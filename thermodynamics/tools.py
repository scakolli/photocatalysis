import os
import matplotlib.pyplot as plt
import numpy as np
import logging

from xtb.ase.calculator import XTB
from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MUTED
from xtb.interface import Calculator, Param
from xtb.utils import get_method as get_xtb_method

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

def single_point(molecule, method="GFN2-xTB", accuracy=0.2, electronic_temperature=298.15, relaxation=False,
                 trajectory=None, fmax=0.05):
    # Attach Caclulator to ASE molecule
    molecule.calc = XTB(method=method, accuracy=accuracy, electronic_temperature=electronic_temperature)
    # Locally optimize geometry
    if relaxation:
        molecule.info['opt_file'] = trajectory
        optimizer = LBFGS(molecule, trajectory=trajectory, logfile=None)
        # logger = get_logger()
        # logger.info('Optimizing Geometry')
        optimizer.run(fmax=fmax)

    # Calculate Energy and delete calc so you can pickle molecule w/o errors
    molecule.info['energy'] = molecule.get_potential_energy()
    del molecule.calc

    return molecule

def single_point_worker(jobs):
    molecule, calc_param_kwargs, opt_param_kwargs = jobs
    return single_point(molecule, **calc_param_kwargs, **opt_param_kwargs)

def HOMO_LUMO_energy(molecule, method='GFN2-xTB', accuracy=0.2, electronic_temperature=298.15):
    """
    Returns HOMO and LUMO energies (eV) of an ASE molecule. Molecule must have an ASE calculator instance already attached.
    Note: Conversion from ASE's use of Angstrom/eV, to xTB's use of Bohr/Hartree
    """
    num, pos = molecule.numbers, molecule.positions / Bohr

    if molecule.calc is None:
        calculator = Calculator(get_xtb_method(method), num, pos)
        calculator.set_accuracy(accuracy)
        calculator.set_electronic_temperature(electronic_temperature)
    else:
        params = molecule.calc.parameters
        calculator = Calculator(get_xtb_method(params['method']), num, pos)
        calculator.set_accuracy(params['accuracy'])
        calculator.set_electronic_temperature(params['electronic_temperature'])

    calculator.set_verbosity(VERBOSITY_MUTED)

    results = calculator.singlepoint()
    occup = results.get_orbital_occupations().astype(int)
    energies = results.get_orbital_eigenvalues()

    homo_indx = np.nonzero(occup)[0][-1]  # Last non-zero orbital occupancy
    lumo_indx = homo_indx + 1

    return energies[homo_indx] * Hartree, energies[lumo_indx] * Hartree

def zero_point_energy(molecule):
    ### Calculate vibrational modes by finite-diff. approximation of the Hessian and return the zero point energy (ZPE)
    displacement = 0.005 * Bohr  # Value used in 'xtb mol.xyz --hess'
    vib = Vibrations(molecule, delta=displacement)
    vib.run(), vib.combine()
    vib_energy = vib.get_energies()
    vib.clean(), os.rmdir('vib')

    if len(molecule) == 2:
        # Hydrogen molecule exception (3N-5 for linear molecule)
        num_modes = 1
    else:
        num_modes = 3 * len(molecule) - 6  # 3N-6 for nonlinear

    zpe = vib_energy[-num_modes:].sum().real / 2.
    return zpe

def calculate_free_energies(s, OH, O, OOH):
    ### Calculate reaction free energies of *, *OH, *O, *OOH species
    if 'zpe' not in OOH.info:
        s.info['zpe'] = zero_point_energy(s)
        OH.info['zpe'] = zero_point_energy(OH)
        O.info['zpe'] = zero_point_energy(O)
        OOH.info['zpe'] = zero_point_energy(OOH)

    Es, EOH, EO, EOOH = s.info['energy'], OH.info['energy'], O.info['energy'], OOH.info['energy']
    ZPEs, ZPEOH, ZPEO, ZPEOOH = s.info['zpe'], OH.info['zpe'], O.info['zpe'], OOH.info['zpe']

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
