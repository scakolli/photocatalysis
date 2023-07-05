import numpy as np
from copy import deepcopy

def F(y):
    # Fitness: 
    # A largre driving potential coupled with a low overpotential, makes for an ideal candidate. Maximize this quantity
    # y.shape : (2, N) array
    IP, dGmax = y
    driving_potential = IP - 1.23
    overpotential = dGmax - 1.23
    return driving_potential - overpotential

def F_acqusition(y, std, kappa, return_array=False):
    # Acquisition Function, Facq = F + k * sigma
    # y.shape, std.shape : (2, N) arrays
    # Pass std = np.array([]) if no std's
    assert y.shape[0] == 2, "Misshaped array"
    utility = F(y)
    var_utility = np.sum(std ** 2, axis=0) # Error Propagation of linear fitness function above...
    std_utility = np.sqrt(var_utility) # ndarray
    Facq = utility + kappa * std_utility

    if not return_array:
        return Facq
    else:
        # I think we should return [utility, std_utility]?
        return np.array([Facq, std_utility]).T