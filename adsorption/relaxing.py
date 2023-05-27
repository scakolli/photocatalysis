import time
from copy import deepcopy
from osc_discovery.photocatalysis.thermodynamics.tools import single_point
from osc_discovery.photocatalysis.adsorption.tools import pairwise, get_neighboring_bonds_list

def relax_configurations(configurations, opt_filenames=None, fmax=0.05):
    ### Relax each configuration within the supplied list
    # Note: Intial configurations have no calculator object. If calculator params other than default desired,
    # first perform a single point calculation with desired params. before relaxation:
    # i.e. [single_point(config, method=desired_method, accuracy=desired_acc, etc.) for config in configurations]

    # Generate optimization output file names
    num_configs = len(configurations)
    if opt_filenames is not None:
        opt_filenames = [f'{opt_filenames}{i}.traj' for i in range(num_configs)]
    else:
        opt_filenames = [None] * num_configs

    # Perform Relaxations
    start = time.perf_counter()
    for i, config in enumerate(configurations):
        config.info['energy'] = single_point(config, relaxation=True, fmax=fmax, trajectory=opt_filenames[i])
        config.info['opt_file'] = opt_filenames[i]

        print('Configuration:', i, '/', num_configs - 1)
        print('Energy (eV):', config.info['energy'])
        print('----------------------------------------')
    print('Elapsed Time:', time.perf_counter() - start)
    print('----------------------------------------')


def check_site_identity_volatilization(composite_relaxed, substrate, volatilization_threshold=2):
    """
    Upon relaxation of a configuration:
    1. Check if identity of substrate has been preserved
    2. Check if identity of adsorbate has been preserved
    3. Check if adsorbate is properly bound and has not volatilized off the surface
    4. Check where the active site is
    """

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