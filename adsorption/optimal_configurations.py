import numpy as np
from copy import deepcopy
import os

from ase.constraints import FixAtoms
from osc_discovery.photocatalysis.adsorption.constants import OH, O, OOH, HOOKEAN_OH, HOOKEAN_OOH_A, HOOKEAN_OOH_B
from osc_discovery.photocatalysis.adsorption.building import build_configuration_from_site
from osc_discovery.photocatalysis.adsorption.relaxing import relax_configurations, filter_configurations
from osc_discovery.photocatalysis.thermodynamics.tools import get_logger


def fixed_nonH_neighbor_indices(atom_index, substrate, free_nonH_neighbors=12):
    # Returns indices of non-hydrogen atoms that are to be frozen during relaxation
    # What is the average non_hydrogenic size of a given moeity thats added via the morphing operations?
    nonH_nearest_neighbors = substrate.get_distances(atom_index, indices=[range(substrate.info['nonH_count'])]).argsort()
    return nonH_nearest_neighbors[free_nonH_neighbors+1:]

def find_optimal_adsorbate_configurations(substrate, h=1.4, optlevel_low='normal', optlevel_high='vtight', multi_process=6):
    ######## 1. OH low-fidelity relaxation and scan ########
    calc_params = substrate.info['calc_params']

    # Generate configurations for each proposed site (site: non-H atom in substrate)
    configsOH = []
    for site in range(substrate.info['nonH_count']):
        # Perhaps introducing constraints here is the best way...
        # constr_list = [HOOKEAN_OH, FixAtoms(indices=fixed_nonH_neighbor_indices(site, substrate))]
        configsOH += build_configuration_from_site(OH, substrate, site, f=h)

    # Relax configurations and keep legitimate ones
    configsOH_relaxed = relax_configurations(configsOH, optlevel_low, calc_params, multi_process=multi_process)
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
        configsO_relaxed = relax_configurations(configsO, optlevel_low, calc_params, multi_process=1)

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
        configsOOH_relaxed = relax_configurations(configsOOH, optlevel_low, calc_params, multi_process=1)

        configsOOH_filtered = filter_configurations(configsOOH_relaxed, substrate)
        configsOOH_filtered_matched = [config for config in configsOOH_filtered if
                                       active_site in config.info['active_sites']]

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

            fully_optimal_configs = relax_configurations(optimal_configs, optlevel_high, calc_params, multi_process=3)
            optimal_configs_filtered = filter_configurations(fully_optimal_configs, substrate)
            optimal_configs_filtered_matched = [config for config in optimal_configs_filtered if
                                                active_site in config.info['active_sites']]
            if len(optimal_configs_filtered_matched) == 3:
                active_site_logger.info(f'Optimal Active Site Found: {active_site}')
                return optimal_configs_filtered_matched
            else:
                continue

    active_site_logger.error('No Suitable Active Site')