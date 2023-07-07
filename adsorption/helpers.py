from itertools import tee
import traceback
import numpy as np
from scipy.optimize import minimize

import openbabel.pybel as pb
from ase.constraints import FixAtoms
from rdkit import Chem
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms

from osc_discovery.cheminformatics.cheminformatics_misc import ase2xyz

# Keukilization errors when attempting to read 'xyz' files... openbabel doesnt always correctly assign double bonds
# to a molecule, since there can be multiple ways of doing so when all you pass to the parser is the positional coords
# of your molecule. For our purposes (get_neighboring_bonds_list()), it shouldn't matter as we are simply trying to calc
# a neighboring bonds list, and since coordinates of nuclei are unaffected by these errors (only electron distribution
# in the lewis structure rdkit representation), which should still get the correct neighbor assignment. ASE has a geometry
# analysis module for this purpose, but rdkit is fast
pb.ob.obErrorLog.SetOutputLevel(0) # Only Critical errors (0)... default Warning (1)

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

def equivalent_atoms_grouped(smile):
    # The canonical rank of each atom in the molecule represents the unique symmetry group it belongs to according to
    # to a 2D-graph representation. In 3D however, each atom could have a unique electronic environment. Output a list
    # of equivalent atom indices.
    rdkit_molecule = Chem.MolFromSmiles(smile)
    can_rank = list(CanonicalRankAtoms(rdkit_molecule, breakTies=False))
    symmetries = {k: [] for k in can_rank}
    for j, r in enumerate(can_rank):
        symmetries[r].append(j)

    return list(symmetries.values())

def equivalent_atoms(eqv_atoms_grouped, charges, symmetry_charge_thresh=0.001):
    # Check for molecular graph symmetry and return a list of equivalent, non-H atoms
    # 2 atoms that are equivalent according to the 2D graph symmetry of the substrate, are not necessarily
    # equivalent in the 3D case... each atom will experience a unique electronic environment in 3D!
    # But the 2D graph equivalent atoms affer an excellent approximation (2D equivalent atoms usually are differing ~0.05 eV
    # in energy from the 3D case, and often times much smaller differences are encountered)
    # The situation where this approximation appears to break down, is when the 3D electronic environments, given
    # say by the partial charges on each atom, are very different from one another (on the order of 0.001 Coloumb)...
    # then you need to consider each atom individually in this case.

    charges_grouped = [[charges[i] for i in symgroup] for symgroup in eqv_atoms_grouped]
    max_q_difference = [max(q) - min(q) for q in charges_grouped]

    eqv_atoms = []
    for e, q in zip(eqv_atoms_grouped, max_q_difference):
        if q < symmetry_charge_thresh:
            # Equiv by symmetry and electronic env. Return any index (in this case 0)
            eqv_atoms.append(e[0])
        else:
            # print('###################################')
            # print('Inequivalent atoms detected based on charge threshold...')
            # print(f'Adding {e}')
            # assert q < symmetry_charge_thresh, "Inequivalent atoms by charge. Debug"
            # Equiv by symmetry, but not by electronic env. Return all indices
            eqv_atoms += [*e]

    return eqv_atoms

def dist_obj(point, points):
    # Closest neighbor distance (negative)
    return -np.linalg.norm(point - points, axis=1).min()

def find_constrained_optimal_position(site_position, sub_position, distance_constraint=1.4):
    """Space out the adsorbate on the active site so its not close to any other atoms"""
    # Given the position of an atom on the substrate, that is proposed to be an active site,
    # minimize the distance of an adsorbate to the site's nearest neighbors, subject to a constraint
    # that the adsorbate isn't a 'distance_constraint' away from the active site.

    initial_guess = site_position

    # Optimization
    bounds = [(None, None), (None, None), (None, None)]
    constraints = [{'type': 'eq',
                    'fun': lambda x: np.linalg.norm(x - site_position) - distance_constraint}]

    result = minimize(dist_obj, initial_guess, args=(sub_position), method='SLSQP', bounds=bounds,
                      constraints=constraints)

    return result.x

def multiprocessing_run_and_catch(multiprocessing_iterator):
    # Improved error handling using multiprocessing.Pool.imap()
    # Supply a multiprocessing iterator. Try to run jobs and catch errors. Return only successfully completed jobs, and caught errors.
    results, errors = [], []
    iteration = 0
    while True:
        try:
            result = next(multiprocessing_iterator)
            results.append(result)
        except StopIteration:
            break
        except Exception as e:
            errors.append((iteration, traceback.format_exc()))
        iteration += 1

    return results, errors

