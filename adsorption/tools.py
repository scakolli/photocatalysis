import numpy as np
from copy import deepcopy
import multiprocessing
from itertools import repeat
import time

from rdkit import Chem
from photocatalysis.conformers import get_conformers_rdkit as get_conformers
from photocatalysis.thermodynamics.tools import single_run
from photocatalysis.thermodynamics.helpers import get_logger, get_multi_process_cores

from photocatalysis.adsorption.helpers import multiprocessing_run_and_catch, get_neighboring_bonds_list, equivalent_atoms_grouped, equivalent_atoms
from photocatalysis.adsorption.helpers import find_constrained_optimal_position
from photocatalysis.adsorption.constants import OH, O, OOH

from photocatalysis.adsorption.helpers import ase2rdkit_valencies
from osc_discovery.cheminformatics.cheminformatics_misc import rdkit2ase
from rdkit.Chem import AllChem

def prepare_substrate(smile_string, calculator_params, multi_process_conf=1, multi_process_sp=1, symmetry_charge_thresh=0.001, job_number=0, print_conf_gen_output=False):
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
    # print(f'Process {multiprocessing.current_process().name} started working on task', flush=True)
    # Generate sorted low-energy conformations using ETKDG and MMFF94
    substrate_confs = get_conformers(smile_string, n_cpu=multi_process_conf, print_output=print_conf_gen_output)
    substrate = substrate_confs.pop(0) # Lowest energy conf

    # Relax at the tight-binding level with xTB, determine zero-point energy and entropy contributions, get charge population
    # analysis for 2D symmetry checking purposes
    substrate = single_run(substrate, runtype='ohess vtight', **calculator_params, pop='', parallel=multi_process_sp, job_number=job_number)
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
    substrate.info['smi'] = smile_string
    del substrate.info['qs']

    return substrate

def prepare_substrate_worker(job):
    job_num, job_input = job
    smile_string, calc_params_dict = job_input
    return prepare_substrate(smile_string, calc_params_dict, job_number=job_num)

def multi_prepare_substrate(smile_string_list, calc_kwargs=None, multi_process=-1):
    # Generate (job_number, (single_run_parameters)) jobs to send to 
    multi_process = get_multi_process_cores(len(smile_string_list), multi_process) # multi_process=-1 returns returns max efficient cores, else return multi_process
    jobs = list(enumerate(zip(smile_string_list, repeat(calc_kwargs))))

    job_logger = get_logger()
    job_logger.info(f'substrate preparation jobs to do: {len(smile_string_list)}')

    start = time.perf_counter()

    # Parallel Relaxation
    with multiprocessing.Pool(multi_process) as pool:
        substrates_iterator = pool.imap(prepare_substrate_worker, jobs)
        substrates_list, substrates_errors = multiprocessing_run_and_catch(substrates_iterator)

    # Attach smile identifiers
    substrates_list = [(smile_string_list[indx], sub) for indx, sub in substrates_list]
    substrates_errors = [(smile_string_list[indx], error) for indx, error in substrates_errors]

    job_logger.info(f'finished jobs. Took {time.perf_counter() - start}s')

    return substrates_list, substrates_errors

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
    config_list = []

    if isinstance(site, list):
        assert len(site) == 2, 'build_config from more than 3 sites...??? Misbehavior'
        # Oxygen bonded to 2 atoms
        p = s[site].positions.mean(axis=0)

        # Pick 1 site
        s0 = site[0]
        p_center = s[s0].position
        bs = s.info['bonds'][s0]
        b = [[b for b in bs if b not in site][0]] + [site[1]] # One non-O- and one O-bonded atom index

        diff = s[b].positions - p_center
        cross = np.cross(diff[0], diff[1])
        n = cross / np.linalg.norm(cross)

        if len(a) > 1:
            a.rotate(a[1].position, n)

        a.translate(p + n * f)

        s.info.clear() #else, you attach s.info to composites and thats unnecessary baggage during computations
        composite = s + a
        config_list.append(composite)

        return config_list

    # Proposed active site position and indices of neighboring bonded atoms
    # where len(b) is the bond order of atom
    p = s[site].position
    b = s.info['bonds'][site]

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