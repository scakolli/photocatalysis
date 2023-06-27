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
    std_utility = np.sqrt(var_utility)
    Facq = utility + kappa * std_utility

    if not return_array:
        return Facq
    else:
        # I think we should return [utility, std_utility]?
        return np.array([Facq, std_utility]).T

    

def get_F(y, ideal_points=[0.0,-5.1], weights=[1.0,0.7],
                           std=[], kappa=1., return_array=False):
    
    ''' Compute utility function F and acquisition function Facq = F + k*\sigma '''

    y[0]/=1000. # lambda to eV
    
    utility_hole = -np.sqrt(( (y[0]-ideal_points[0]) * 1.0)**2 + ( (y[1]-ideal_points[1]) * 0.7)**2) 
    
    if len(std)>0:
        
        # for compatiblity with order in AL 
        std[0]/=1000.
        var_hole = np.sum( utility_hole**(-2) * np.array([weights[0], weights[1]])**4 * \
                          (np.array([ y[0], y[1] ]) - np.array( [ideal_points[0],ideal_points[1]] ))**2 * \
                           np.array( [std[0],std[1]] )**2 )
        
        std_hole = np.sqrt(var_hole)
        utility_hole += kappa*std_hole

        if return_array:
            return [ utility_hole, std_hole ]
    
    if return_array:
        return [ utility_hole, 0 ]
    
    return utility_hole
