from cheminformatics_fingerprint import get_tanimoto_distmat_multiprocessing
import pickle
# import os
import sys
import numpy as np

if __name__ == '__main__':
    dist_info = sys.argv[1]

    with open(dist_info, 'rb') as f:
        X1, X2, dmat_loc, multiprocess = pickle.load(f)

    D = get_tanimoto_distmat_multiprocessing(X1, X2, multiprocess=multiprocess)
    
    print(dmat_loc, multiprocess)
    np.save(dmat_loc, D)
