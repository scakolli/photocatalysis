import numpy as np
from ase.constraints import FixAtoms
from osc_discovery.photocatalysis.adsorption.constants import OH, O, OOH, HOOKEAN_OH, HOOKEAN_OOH_A, HOOKEAN_OOH_B
from osc_discovery.photocatalysis.adsorption.building import build_configuration_from_site
from osc_discovery.photocatalysis.adsorption.relaxing import relax_configurations, filter_configurations


def fixed_nonH_neighbor_indices(atom_index, substrate, free_nonH_neighbors=12):
    # Returns indices of non-hydrogen atoms that are to be frozen during relaxation
    # What is the average non_hydrogenic size of a given moeity thats added via the morphing operations?
    nonH_nearest_neighbors = substrate.get_distances(atom_index, indices=[range(substrate.info['nonH_count'])]).argsort()
    return nonH_nearest_neighbors[free_nonH_neighbors+1:]

def find_optimal_adsorbate_configurations(substrate, h=1.4, opt_logs=None):

    if opt_logs is not None:
        opt_filenames = [f'{opt_logs}OH', f'{opt_logs}O', f'{opt_logs}OOH']
    else:
        opt_filenames = [None] * 3

    ######## 1. OH low-fidelity relaxation and scan ########

    # Generate configurations for each proposed site (site: non-H atom in substrate)
    configsOH = []
    for site in range(substrate.info['nonH_count']):
        constr_list = [HOOKEAN_OH, FixAtoms(indices=fixed_nonH_neighbor_indices(site, substrate))]
        configsOH += build_configuration_from_site(OH, substrate, site, constr_list, f=h)

    # Relax configurations and keep legitimate ones
    relax_configurations(configsOH, opt_filenames=opt_filenames[0])
    configsOH_filtered = filter_configurations(configsOH, substrate)

    # Rank active sites by adsorption energy
    energies = np.array([config.info['energy'] for config in configsOH_filtered])
    active_sites_configsOH = np.array([config.info['active_sites'][0] for config in configsOH_filtered])

    indx_sorted = energies.argsort()
    active_sites_ranked = active_sites_configsOH[indx_sorted]

    ######## 2. O and OOH relaxation ########

    for j, _active_site in enumerate(active_sites_ranked):
        active_site = int(_active_site)

        print('Testing Active Site:', active_site)
        print('----------------------------------------')

        ### O intermediate
        # For candidate low-energy active site, build O* configs and relax
        constrO = [FixAtoms(indices=fixed_nonH_neighbor_indices(active_site, substrate))]
        configsO = build_configuration_from_site(O, substrate, active_site, constrO, f=h)
        relax_configurations(configsO, opt_filenames=opt_filenames[1])

        # Filter for legitimate configurations and active site matches
        configsO_filtered = filter_configurations(configsO, substrate)
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
        constrOOH = [HOOKEAN_OOH_A, HOOKEAN_OOH_B,
                     FixAtoms(indices=fixed_nonH_neighbor_indices(active_site, substrate))]
        configsOOH = build_configuration_from_site(OOH, substrate, active_site, constrOOH, f=h)
        relax_configurations(configsOOH, opt_filenames=opt_filenames[2])

        configsOOH_filtered = filter_configurations(configsOOH, substrate)
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
            del optimal_OHconfig.constraints, optimal_Oconfig.constraints, optimal_OOHconfig.constraints
            optimal_configs = [optimal_OHconfig, optimal_Oconfig, optimal_OOHconfig]
            relax_configurations(optimal_configs, fmax=0.005)
            optimal_configs_filtered = filter_configurations(optimal_configs, substrate)
            optimal_configs_filtered_matched = [config for config in optimal_configs_filtered if
                                                active_site in config.info['active_sites']]
            if len(optimal_configs_filtered_matched) == 3:
                print('----------------------------------------')
                print('----------------------------------------')
                print("Optimal Active Site Found:", active_site)
                break
            else:
                continue

    return optimal_configs_filtered_matched