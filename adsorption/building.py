import numpy as np
from scipy.optimize import minimize

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

def build_configuration_from_site(adsorbate, substrate, site, constraint_list, f=1.4):
    """Build a list of rudimentary adsorbate/substrate configurations on a proposed active site"""
    a, s = adsorbate.copy(), substrate.copy()

    # Proposed active site position and indices of neighboring bonded atoms
    # where len(b) is the bond order of atom
    p = s[site].position
    b = s.info['bonds'][site]
    config_list = []

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

        # Form composite system and introduce any constraints
        composite = s + a
        composite.set_constraint(constraint_list)
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
            composite_opt.set_constraint(constraint_list)
            config_list.append(composite_opt)

    return config_list