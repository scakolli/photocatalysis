import numpy as np
from copy import deepcopy

from rdkit import Chem
from osc_discovery.descriptor_calculation.conformers import get_conformers_rdkit as get_conformers
from photocatalysis.thermodynamics.tools import single_run

from photocatalysis.adsorption.helpers import get_neighboring_bonds_list, equivalent_atoms_grouped, equivalent_atoms
from photocatalysis.adsorption.helpers import find_constrained_optimal_position
from photocatalysis.adsorption.constants import OH, O, OOH

from photocatalysis.adsorption.helpers import ase2rdkit_valencies
from osc_discovery.cheminformatics.cheminformatics_misc import rdkit2ase
from rdkit.Chem import AllChem

def prepare_substrate(smile_string, calculator_params, multi_process_conf=1, multi_process_sp=1, symmetry_charge_thresh=0.001, return_all=False):
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
    substrate_confs = get_conformers(smile_string, n_cpu=multi_process_conf)
    substrate = substrate_confs.pop(0) # Lowest energy conf

    # Relax at the tight-binding level with xTB, determine zero-point energy and entropy contributions, calculate
    # the ionization potential, and then request a charge population analysis for 2D symmetry checking purposes
    substrate = single_run(substrate, runtype='ohess vtight', **calculator_params, parallel=multi_process_sp)
    substrate = single_run(substrate, runtype='vipea', **calculator_params, pop='', parallel=multi_process_sp)
    qs = substrate.info['qs']  # charges

    # Attach useful information to the substrate object
    # total_num_nonHs = len(substrate) - substrate.get_chemical_symbols().count('H')  # number of non-hydrogen atoms
    nonH_atoms = [atom.index for atom in substrate if atom.symbol != 'H']
    eqv_atoms_grp = equivalent_atoms_grouped(smile_string)

    substrate.info['equivalent_atoms_grouped'] = eqv_atoms_grp
    substrate.info['equivalent_atoms'] = equivalent_atoms(eqv_atoms_grp, qs, symmetry_charge_thresh=symmetry_charge_thresh)
    substrate.info['calc_params'] = deepcopy(calculator_params)
    substrate.info['bonds'] = get_neighboring_bonds_list(substrate)
    substrate.info['nonH_count'] = len(nonH_atoms)
    del substrate.info['qs']

    if return_all:
        return substrate_confs
    else:
        return substrate

def get_adsorbate_conformers(molecule, numConfs=10, numThreads=4, return_all=False):
    ### Generate conformers using RDKIT and optimize them with a FF. Sorted by energy.
    molecule_copy = deepcopy(molecule)
    mol = ase2rdkit_valencies(molecule_copy)
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=numThreads)
    ff_opt_energies = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=numThreads)

    conformers = [rdkit2ase(mol, confId=j) for j in range(mol.GetNumConformers())]
    conformers_sorted = [c for _, c in sorted(zip(ff_opt_energies, conformers), key=lambda x: x[0][1])]

    if return_all:
        return conformers_sorted
    else:
        return conformers_sorted[0]

def build_configuration_from_site(adsorbate, substrate, site, f=1.4):
    """Build a list of rudimentary adsorbate/substrate configurations on a proposed active site"""
    a, s = adsorbate.copy(), substrate.copy()

    # Proposed active site position and indices of neighboring bonded atoms
    # where len(b) is the bond order of atom
    p = s[site].position
    b = s.info['bonds'][site]
    config_list = []

    if (s[site].symbol == 'N') and (len(b) == 3):
        # Ignore Tertiary Nitrogens... specifically the Hbonded N's need to not be considered....
        return config_list
    # Alkane carbon (4) or Carbonyl Oxygen (1), likely not active sites... skip
    # 1. Create a configuration where adsorbate is on-top of site (perpendicular to plane formed by its neighbors)
    # 2. Create an additional heteroatom configuration, by maximizing distance of adsorb. relative to neighbors
    if (len(b) != 4) & (len(b) != 1):

        # Determine vector 'n' that defines a plane between active site and 2 neighboring atoms
        diff = s[b].positions - p
        cross = np.cross(diff[0], diff[1])
        n = cross / np.linalg.norm(cross)

        # Rotate vector definining O-H or O-O bond into normal vector 'n'
        if len(a) > 1:
            a.rotate(a[1].position, n)

        # Translate adsorbate a height 'f' above position of active site
        a.translate(p + n * f)

        # Form composite system
        s.info.clear() #else, you attach s.info to composites and thats unnecessary baggage during computations
        composite = s + a
        config_list.append(composite)

        if s[site].symbol in ['O', 'N', 'S']:
            a_opt = adsorbate.copy()
            other_positions = np.delete(s.positions, site, axis=0)  # Remove active site iteslf from position array
            optimized_adsorb_position = find_constrained_optimal_position(p, other_positions, distance_constraint=f)
            n_orient = optimized_adsorb_position - p

            if len(a_opt) > 1:
                a_opt.rotate(a_opt[1].position, n_orient)

            a_opt.translate(optimized_adsorb_position)

            composite_opt = s + a_opt
            config_list.append(composite_opt)

    return config_list

def build_configurations(substrate, sites, height=1.4):
    configsOH, configsO, configsOOH = [], [], []
    for site in sites:
        configsOH += build_configuration_from_site(OH, substrate, site, f=height)
        configsO += build_configuration_from_site(O, substrate, site, f=height)
        configsOOH += build_configuration_from_site(OOH, substrate, site, f=height)

    return configsOH, configsO, configsOOH