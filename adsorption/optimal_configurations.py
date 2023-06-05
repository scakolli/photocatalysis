import numpy as np
from copy import deepcopy
import os

from osc_discovery.photocatalysis.adsorption.tools import pairwise, get_neighboring_bonds_list
from osc_discovery.photocatalysis.adsorption.constants import OH, O, OOH, HOOKEAN_OH, HOOKEAN_OOH_A, HOOKEAN_OOH_B
from osc_discovery.photocatalysis.adsorption.building import build_configuration_from_site
from osc_discovery.photocatalysis.thermodynamics.tools import multi_run, get_logger

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
    bonds = get_neighboring_bonds_list(c)
    b = deepcopy(bonds)
    bonds_dict = dict(zip(range(len(b)), b))

    ### 4. Active-site Check
    # Where is the Oxygen in the adsorbate bonded
    sites = [sub_indx for sub_indx in bonds[ads_indx[0]] if sub_indx not in ads_indx]

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
    # If min distance from adsorbate to substrate is beyond the volatalization threshold, condition fails
    # Hydrogen bonding distance ~ 2.5 Angstroms, for reference
    # There is perhaps a faster way to do this using only the 'bonds' list
    # if len(sites) == 0, then volatization has occured... i think rdkit might be good enough to tell when volatization has happened
    min_dist = c.get_distances(ads_indx[0], indices=[i for i in range(s.info['nonH_count'])]).min()
    cond3 = (min_dist < volatilization_threshold)

    return sites, all([cond1, cond2, cond3])

def filter_configurations(configurations, substrate):
    ### Perform fidelity checks on a list of configurations, attach active site info, and filter
    filtered_configs = []
    for config in configurations:
        config.info['active_sites'], checks = check_site_identity_volatilization(config, substrate)

        if checks: filtered_configs.append(config)

    return filtered_configs

def find_optimal_adsorbate_configurations(substrate, h=1.4, optlevel_low='normal', optlevel_high='vtight', keep_folders=False, multi_process=6):
    ######## 1. OH low-fidelity relaxation and scan ########
    calc_params = substrate.info['calc_params'].copy()

    # Generate configurations for each proposed site (site: non-H atom in substrate)
    configsOH = []
    for site in range(substrate.info['nonH_count']):
        # Perhaps introducing constraints here is the best way...
        # constr_list = [HOOKEAN_OH, FixAtoms(indices=fixed_nonH_neighbor_indices(site, substrate))]
        configsOH += build_configuration_from_site(OH, substrate, site, f=h)

    return configsOH
    # Relax configurations and keep legitimate ones
    if keep_folders: os.mkdir('OH'), os.chdir('OH')
    configsOH_relaxed = multi_run(configsOH, runtype=f'opt {optlevel_low}', keep_folders=keep_folders,
                                  calc_kwargs=calc_params, multi_process=multi_process)
    configsOH_filtered = filter_configurations(configsOH_relaxed, substrate)

    # Rank active sites by adsorption energy
    energies = np.array([config.info['energy'] for config in configsOH_filtered])
    active_sites_configsOH = np.array([config.info['active_sites'][0] for config in configsOH_filtered])
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