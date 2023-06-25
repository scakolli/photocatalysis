import numpy as np
from copy import deepcopy
import os
from ase.geometry.analysis import Analysis
import itertools

from photocatalysis.adsorption.helpers import pairwise, get_neighboring_bonds_list
from photocatalysis.adsorption.constants import OH, O, OOH
from photocatalysis.adsorption.tools import build_configurations
from photocatalysis.thermodynamics.tools import multi_run
from photocatalysis.thermodynamics.helpers import create_trajectories_from_logs

def build_and_relax_configurations(substrate, sites, optlevel='loose', multi_process=-1, additional_conformers=False):
    ### Build Configs
    configsOH, configsO, configsOOH = build_configurations(substrate, sites)
    num_configs = len(configsOH) # num configs each, independent of adsorbate

    ### Relaxation
    calc_kwargs_sub = deepcopy(substrate.info['calc_params']) 
    configs = multi_run(configsOH+configsO+configsOOH, runtype=f'opt {optlevel}', calc_kwargs=calc_kwargs_sub, multi_process=multi_process)
    configsoh, configso, configsooh = np.array(configs, dtype='object').reshape(3, num_configs).tolist()

    ### Generate additional conformers using ETKGT and FF's, and relax
    if additional_conformers:
        pass

    ### Filtering
    configsoh = filter_configurations(configsoh, substrate)
    configso = filter_configurations(configso, substrate)
    configsooh = filter_configurations(configsooh, substrate)

    return configsoh, configso, configsooh

def check_site_identity_volatilization(composite_relaxed, substrate, volatilization_threshold=2):
    """
    Upon relaxation of a configuration:
    1. Check if identity of substrate has been preserved
    2. Check if identity of adsorbate has been preserved
    3. Check if adsorbate is properly bound and has not volatilized off the surface
    4. Check where the active site is
    """
    assert len(composite_relaxed) != len(substrate), "No adsorbate attached"

    c, s = composite_relaxed.copy(), substrate.copy()
    ads_indx = [indx for indx in range(len(s), len(c))]  # adsorbate indices

    # List of bond indices for each atom, arranged in ascending order. Create dict for easy processing.
    # RDKIT method with get_neighboring_bonds_list() is much faster at assigning bonds, but it fails
    # to correctly classify adsorption bonds sometimes... ASE might be better suited in this regard, even if
    # it is slow in assigning bonds

    # bonds = get_neighboring_bonds_list(c) # RDKIT
    bonds = Analysis(c).all_bonds[0] # ASE
    b = deepcopy(bonds)
    bonds_dict = dict(zip(range(len(b)), b))

    ### 4. Active-site Check
    # Where is the Oxygen in the adsorbate bonded.
    sites = [bonded_atom_index for bonded_atom_index in bonds[ads_indx[0]] if bonded_atom_index not in ads_indx]

    ### 1. Substrate Check
    # Remove the adsorbate from the bonds dict and compare the resulting dict to the original substrate bonds list
    # Look at what the adsorbate (a_i) is bonded to (v), and remove the adsorbate from v's bonds list
    # Then delete the adsorbate from the dict
    for a_i in ads_indx:
        for v in bonds_dict[a_i]:
            bonds_dict[v].remove(a_i)
        del bonds_dict[a_i]

    cond1 = (list(bonds_dict.values()) == s.info['bonds'])

    ### 2. Adsorbate Check
    # Define equil. OH and OO bond distances, threshold for bond-breakage
    equilibrium_distances, threshold = [0.963, 1.302], 0.3
    ads_dist = [c.get_distance(*ai) for ai in pairwise(ads_indx)]
    ads_dist.reverse()
    cond2 = all([(abs(i - j) < threshold) for i, j in zip(equilibrium_distances, ads_dist)])

    ### 3. Volatilization Check
    # (a) If min distance from adsorbate to substrate is beyond the volatalization threshold, condition fails
    # (b) ASE has covalent radii as reference (and vdW radii), and it makes bond assesments based on that
    # Since we are interested in chemisorbed configurations that lead to relatively large binding energies,
    # it makes sense that adsorption radii are on the order of covalent radii. Max covalent bond length for chemisorption
    # in our system is the S-O bond (~1.71 Angstrom). Instead of imposing a manual volatization threshold, determine
    # volatization from 'len(sites)'.
    # Note: If an H is abstracted from the substrate (*-H + OH -> *+ + H2O), then len(sites) != 0, volitazion condition
    # passes, but substrate condition fails, so the configuration doesn't pass the filter.
    cond3 = (len(sites) != 0)

    # min_dist = c.get_distances(ads_indx[0], indices=[i for i in range(s.info['nonH_count'])]).min()
    # cond3 = (min_dist < volatilization_threshold)

    return sites, all([cond1, cond2, cond3])

def filter_configurations(configurations, substrate):
    ### Perform fidelity checks on a list of configurations, attach active site info, and filter for checks and duplicate sites
    config_dict = dict()

    for j, config in enumerate(configurations):
        sites, checks = check_site_identity_volatilization(config, substrate)
        if checks:
            if len(sites) == 2:
                if repr(sites) not in config_dict:
                    # New site, add config to dict
                    config_dict[repr(sites)] = config
                elif config.info['energy'] < config_dict[repr(sites)].info['energy']:
                    # Config already exists in dict. Replace config with lower energy
                    config_dict[repr(sites)] = config

            elif len(sites) == 1:
                if sites[0] not in config_dict:
                    # New site, add config to dict
                    config_dict[sites[0]] = config
                elif config.info['energy'] < config_dict[sites[0]].info['energy']:
                    # Replace config with lower energy one at a given site
                    config_dict[sites[0]] = config
        else:
            continue

    filtered_configs = []
    for k, config in config_dict.items():
        config = deepcopy(config)
        config.info['active_site'] = k
        filtered_configs.append(config)

    return filtered_configs
