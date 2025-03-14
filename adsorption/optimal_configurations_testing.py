import numpy as np
from copy import deepcopy
import os
from ase.geometry.analysis import Analysis
import itertools

from osc_discovery.photocatalysis.adsorption.helpers import pairwise, get_neighboring_bonds_list
from osc_discovery.photocatalysis.adsorption.constants import OH, O, OOH, HOOKEAN_OH, HOOKEAN_OOH_A, HOOKEAN_OOH_B
from osc_discovery.photocatalysis.adsorption.tools import build_configuration_from_site
from osc_discovery.photocatalysis.thermodynamics.tools import multi_run, get_logger
from osc_discovery.photocatalysis.thermodynamics.helpers import create_trajectories_from_logs

def relax_all_adsorbate_configurations(substrate, calc_params, h=1.4, optlevel='normal', keep_folders=False, multi_process=6):
    relax_logger = get_logger()

    ################ 1. OH relaxation ################
    if keep_folders: os.mkdir('OH'), os.chdir('OH')
    relax_logger.info('OH active site scan')

    # Generate configurations for each proposed site (site: non-H atom in substrate)
    configsOH = []
    for site in range(substrate.info['nonH_count']):
        # Perhaps introducing constraints here is the best way...
        # constr_list = [HOOKEAN_OH, FixAtoms(indices=fixed_nonH_neighbor_indices(site, substrate))]
        configsOH += build_configuration_from_site(OH, substrate, site, f=h)

    # Relax configurations
    configsOH_relaxed = multi_run(configsOH, runtype=f'opt {optlevel}', keep_folders=keep_folders,
                                  calc_kwargs=calc_params, multi_process=multi_process)

    # Keep legitimate ones
    configsOH_filtered = filter_configurations(configsOH_relaxed, substrate)

    ################ 2. O relaxation ################
    if keep_folders: os.chdir('..'), os.makedirs('O', exist_ok=True), os.chdir('O')
    relax_logger.info('O active site scan')

    configsO = []
    for site in range(substrate.info['nonH_count']):
        # constrO = [FixAtoms(indices=fixed_nonH_neighbor_indices(active_site, substrate))]
        configsO += build_configuration_from_site(O, substrate, site, f=h)

    configsO_relaxed = multi_run(configsO, runtype=f'opt {optlevel}', keep_folders=keep_folders,
                              calc_kwargs=calc_params, multi_process=multi_process)

    configsO_filtered = filter_configurations(configsO_relaxed, substrate)

    ################ 2. OOH relaxation ################
    if keep_folders: os.chdir('..'), os.makedirs('OOH', exist_ok=True), os.chdir('OOH')
    relax_logger.info('OOH active site scan')

    configsOOH = []
    for site in range(substrate.info['nonH_count']):
        # constrOOH = [HOOKEAN_OOH_A, HOOKEAN_OOH_B, FixAtoms(indices=fixed_nonH_neighbor_indices(active_site, substrate))]
        configsOOH += build_configuration_from_site(OOH, substrate, site, f=h)

    configsOOH_relaxed = multi_run(configsOOH, runtype=f'opt {optlevel}', keep_folders=keep_folders,
                              calc_kwargs=calc_params, multi_process=multi_process)
    configsOOH_filtered = filter_configurations(configsOOH_relaxed, substrate)

    os.chdir('..')
    if keep_folders:
        create_trajectories_from_logs(os.getcwd()) # create .traj files in dirs

    return configsOH_filtered, configsO_filtered, configsOOH_filtered


def find_optimal_adsorbate_configurations_sequential(substrate, h=1.4, optlevel_low='normal', optlevel_high='vtight', keep_folders=False, multi_process=6):
    ######## 1. OH low-fidelity relaxation and scan ########
    calc_params = substrate.info['calc_params'].copy()

    # Generate configurations for each proposed site (site: non-H atom in substrate)
    configsOH = []
    for site in range(substrate.info['nonH_count']):
        # Perhaps introducing constraints here is the best way...
        # constr_list = [HOOKEAN_OH, FixAtoms(indices=fixed_nonH_neighbor_indices(site, substrate))]
        configsOH += build_configuration_from_site(OH, substrate, site, f=h)

    # Relax configurations and keep legitimate ones
    if keep_folders: os.mkdir('OH'), os.chdir('OH')
    configsOH_relaxed = multi_run(configsOH, runtype=f'opt {optlevel_low}', keep_folders=keep_folders,
                                  calc_kwargs=calc_params, multi_process=multi_process)
    configsOH_filtered = filter_configurations(configsOH_relaxed, substrate)

    # Rank active sites by adsorption energy
    energies = np.array([config.info['energy'] for config in configsOH_filtered])
    active_sites_configsOH = np.array([config.info['active_sites'] for config in configsOH_filtered])
    indx_sorted = energies.argsort()
    active_sites_ranked = active_sites_configsOH[indx_sorted]

    active_site_logger = get_logger()
    ######## 2. O and OOH relaxation ########
    for j, _active_site in enumerate(active_sites_ranked):
        active_site = int(_active_site)
        active_site_logger.info(f'Testing Active Site {active_site}')

        ### O intermediate
        # For candidate low-energy active site, build O* configs and relax
        # constrO = [FixAtoms(indices=fixed_nonH_neighbor_indices(active_site, substrate))]
        configsO = build_configuration_from_site(O, substrate, active_site, f=h)

        if keep_folders: os.chdir('..'), os.makedirs('O', exist_ok=True), os.chdir('O')
        configsO_relaxed = multi_run(configsO, runtype=f'opt {optlevel_low}', keep_folders=keep_folders,
                                  calc_kwargs=calc_params, multi_process=1)

        # Filter for legitimate configurations and active site matches
        configsO_filtered = filter_configurations(configsO_relaxed, substrate)
        configsO_filtered_matched = [config for config in configsO_filtered if
                                     active_site in config.info['active_sites']]

        if not len(configsO_filtered_matched):
            # No configs meet filtering criteria, skip to the next active site
            continue
        else:
            # Find min energy configuration and calculate desired properties
            energies = np.array([config.info['energy'] for config in configsO_filtered_matched])
            optimal_Oconfig = configsO_filtered_matched[energies.argmin()]

        ### OOH intermediate
        # constrOOH = [HOOKEAN_OOH_A, HOOKEAN_OOH_B,
        #              FixAtoms(indices=fixed_nonH_neighbor_indices(active_site, substrate))]
        configsOOH = build_configuration_from_site(OOH, substrate, active_site, f=h)

        if keep_folders: os.chdir('..'), os.makedirs('OOH', exist_ok=True), os.chdir('OOH')
        configsOOH_relaxed = multi_run(configsOOH, runtype=f'opt {optlevel_low}', keep_folders=keep_folders,
                                  calc_kwargs=calc_params, multi_process=1)

        configsOOH_filtered = filter_configurations(configsOOH_relaxed, substrate)
        configsOOH_filtered_matched = [config for config in configsOOH_filtered if
                                       active_site in config.info['active_sites']]

        if keep_folders:
            os.chdir('..')
            create_trajectories_from_logs(os.getcwd()) # create .traj files in dirs

        if not len(configsOOH_filtered_matched):
            continue
        else:
            energies = np.array([config.info['energy'] for config in configsOOH_filtered_matched])

            optimal_OOHconfig = configsOOH_filtered_matched[energies.argmin()]
            optimal_OHconfig = configsOH_filtered[indx_sorted[j]]

            # Remove fixed atom and hookean constraints
            # and do a final deep-relaxation and fidelity check
            # del optimal_OHconfig.constraints, optimal_Oconfig.constraints, optimal_OOHconfig.constraints
            optimal_configs = [optimal_OHconfig, optimal_Oconfig, optimal_OOHconfig]

            fully_optimal_configs = multi_run(optimal_configs, runtype=f'opt {optlevel_high}', keep_folders=False,
                                  calc_kwargs=calc_params, multi_process=3)
            optimal_configs_filtered = filter_configurations(fully_optimal_configs, substrate)
            optimal_configs_filtered_matched = [config for config in optimal_configs_filtered if
                                                active_site in config.info['active_sites']]

            if len(optimal_configs_filtered_matched) == 3:
                active_site_logger.info(f'Optimal Active Site Found: {active_site}')
                return optimal_configs_filtered_matched
            else:
                continue

    active_site_logger.error('No Suitable Active Site')

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

    # Return a site and whether or not conditions pass
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

                else:
                    if config.info['energy'] < config_dict[repr(sites)].info['energy']:
                        # Replace config with lower energy one at a given site
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

def filter_configurations_with_symm(configurations, substrate):
    d = dict()

    for config in configurations:
        actv, checks = check_site_identity_volatilization(config, substrate)
        eqv_atoms_grouped = deepcopy(substrate.info['equivalent_atoms_grouped'])

        if checks:
            # Config has not been destroyed/volitalized
            if len(actv) == 2:
                # Remove symmetry duplicate configs
                a0, a1 = actv[0], actv[1]
                a0_equivalents = [symgroup for symgroup in eqv_atoms_grouped if a0 in symgroup][0]
                a1_equivalents = [symgroup for symgroup in eqv_atoms_grouped if a1 in symgroup][0]
                equivalent_active_sites = [sorted(sites) for sites in itertools.product(a0_equivalents, a1_equivalents)]
                equivalent_active_sites.remove(actv) #important

                sites_in_dict = [equivalent for equivalent in equivalent_active_sites if repr(equivalent) in d]

                if not len(sites_in_dict):
                    # No symmetry equivalent active sites present in dict...
                    # Remove duplicate configs, keeping lowest energy one
                    if repr(actv) not in d:
                        d[repr(actv)] = config
                    elif d[repr(actv)].info['energy'] < config.info['energy']:
                        d[repr(actv)] = config
                else:
                    # Equivalent sites present... do not add config
                    pass

            elif len(actv) == 1:
                a = actv[0]
                equivalent_active_sites = [symgroup for symgroup in eqv_atoms_grouped if a in symgroup][0]
                sites_in_dict = [equivalent for equivalent in equivalent_active_sites if equivalent in d]

                if not len(sites_in_dict):
                    #print('symmetry passed')
                    if a not in d:
                        d[a] = config
                    elif d[a].info['energy'] < config.info['energy']:
                        #print('energy updated', a, d[a].info['energy'], config.info['energy'])
                        d[a] = config
                else:
                    pass
                ## if len(actv) == 0:
                    # Volatialized... but on initial relaxation, adsorbate could briefly volatalize before coming back?

    filtered_configs = []
    for k, config in d.items():
        config = deepcopy(config)
        config.info['active_site'] = k
        filtered_configs.append(config)

    return filtered_configs