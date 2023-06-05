from itertools import tee
from copy import deepcopy

from rdkit import Chem
import openbabel.pybel as pb
from ase.constraints import FixAtoms

from osc_discovery.cheminformatics.cheminformatics_misc import ase2xyz
from osc_discovery.descriptor_calculation.conformers import get_conformers_rdkit as get_conformers
from osc_discovery.photocatalysis.thermodynamics.tools import single_run


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

def fixed_nonH_neighbor_indices(atom_index, substrate, free_nonH_neighbors=12):
    # Returns indices of non-hydrogen atoms that are to be frozen during relaxation
    # What is the average non_hydrogenic size of a given moeity thats added via the morphing operations?
    nonH_nearest_neighbors = substrate.get_distances(atom_index, indices=[range(substrate.info['nonH_count'])]).argsort()
    return nonH_nearest_neighbors[free_nonH_neighbors+1:]

def prepare_substrate(smile_string, calculator_params):
    """Prepare the substrate for further use
    get_conformers() generates a list of confs using ETKDG implemented in RDKIT. Depending on the number of
    rotatable bonds, this procedure can create as many as 300 conformations (all satisfying the RMSD pruning criteria
    set in the code). These confs are subsequently optmizied using RDKITs UFF or MFF94, and ranked by energy, and again
    pruned. Lowest energy conformation first in the list.

    Computational High-throughput screen of polymeric photocatalysts
    10.1039/c8fd00171e

    "From a computational high-throughput screening perspective, the observed
    low sensitivity to the sampling of conformational degrees of freedom implies that
    the effect of not finding the true lowest energy conformer on the predicted
    thermodynamic driving force for proton reduction and water oxidation, as well as
    on the on-set of light absorption, is only very minor. Hence a minimal conformer
    search will generally suffice when screening for polymeric photocatalysts. The
    same weak dependence of IP, EA and optical gap values probably also means that
    in contrast to chain length and order/disorder in the case of random co-polymers
    (see below) conformational degrees of freedom do not result in large batch-to-batch variations.

    Maximum variation of a given property with respect to conformation is generally of the order of ~0.1 eV"

    For now, just take lowest energy conformation of the screening procedure, and subsequently optimize with xTB.

    """
    # Generate sorted low-energy conformations using ETKDG and MMFF94
    substrate_confs = get_conformers(smile_string)
    substrate = substrate_confs.pop(0) # Lowest energy conf

    # Relax at the tight-binding level with xTB and determine zero-point energy (parallel single_point calcs)
    substrate = single_run(substrate, runtype='opt vtight', **calculator_params, parallel=4)
    substrate = single_run(substrate, runtype='hess', **calculator_params, parallel=4)

    # Attach useful information to the substrate object
    total_num_nonHs = len(substrate) - substrate.get_chemical_symbols().count('H')  # number of non-hydrogen atoms
    substrate.info['calc_params'] = deepcopy(calculator_params)
    substrate.info['bonds'] = get_neighboring_bonds_list(substrate)
    substrate.info['nonH_count'] = int(total_num_nonHs)

    return substrate, substrate_confs