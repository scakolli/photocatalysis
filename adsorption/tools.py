from itertools import tee
from copy import deepcopy
from rdkit import Chem
import openbabel.pybel as pb

from osc_discovery.cheminformatics.cheminformatics_misc import ase2xyz
from osc_discovery.descriptor_calculation.conformers import get_conformers_rdkit as get_conformers
from osc_discovery.photocatalysis.thermodynamics.tools import single_point


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def ase2rdkit_valencies(atoms, removeHs=False):
    """ Convert an ASE atoms object to rdkit molecule. The ordering of the Atoms is identical."""

    # Updated from previous function to skip the "sanitization" of valence count
    # Ex. Nitrogen cannot normally have an explicit valence of 4, so when an O radical
    # binds to N's within the substrate, sanitization will throw an error since the
    # valence of N is greater than permitted by rdkit

    a_str = ase2xyz(atoms)
    pymol = pb.readstring("xyz", a_str)
    mol = pymol.write("mol")
    mol = Chem.MolFromMolBlock(mol, removeHs=removeHs, sanitize=False)

    # Sanitize everything except for the valencies of the atoms
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)

    return mol

def get_neighboring_bonds_list(substrate):
    # Convert to rdkit molecule and then get bonds
    return [sorted([nbr.GetIdx() for nbr in atom.GetNeighbors()]) for atom in ase2rdkit_valencies(substrate).GetAtoms()]

def prepare_substrate(smile_string, calculator_params):
    # Get molecular conformers of the substrate
    substrate_confs = get_conformers(smile_string)
    substrate = substrate_confs.pop()

    # Relax
    substrate = single_point(substrate, **calculator_params, relaxation=True, fmax=0.005)

    # Attach useful information to the substrate object
    total_num_nonHs = len(substrate) - substrate.get_chemical_symbols().count('H')  # number of non-hydrogen atoms
    substrate.info['calc_params'] = deepcopy(calculator_params)
    substrate.info['bonds'] = get_neighboring_bonds_list(substrate)
    substrate.info['nonH_count'] = int(total_num_nonHs)

    return substrate, substrate_confs