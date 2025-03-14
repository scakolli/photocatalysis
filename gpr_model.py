import time,os,sys
sys.path.append('..')
import numpy as np
import time
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from photocatalysis.cheminformatics_fingerprint import get_tanimoto_distmat
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
import pickle

""" Implements GPR model in simple classes """

class Kernel_method():
    """ A generic Kernel-method base-class """
    
    def __init__(self, constant_kernel=1.0, gamma_kernel=1., sigma_white=0.5, input_type="soap", use_custom=True, multiprocess=1, D_scratch_dir="scratch_distance_matrix"):
        self.input_type = input_type
        self.set_kernel_params(constant_kernel, gamma_kernel, sigma_white)
        self.use_custom = use_custom
        self.multiprocess = multiprocess
        self.X_train=[]
        D=[]
        self.D_scratch_dir = D_scratch_dir
        if not os.path.isdir(self.D_scratch_dir): os.mkdir(self.D_scratch_dir)
        self.normalize_y = True

        print('Number of cores to be used:', self.multiprocess)
        
    def set_kernel_params(self, constant_kernel, gamma_kernel, sigma_white):
        self.constant_kernel=constant_kernel
        self.gamma_kernel=gamma_kernel
        self.sigma_white=sigma_white
        
    # def distance_matrix(self, X1, X2=[], matrix_dest_loc=None):
    #     if self.multiprocess == 1:
    #         print('Single Processing')
    #         return get_tanimoto_distmat(X1, X2)    
    #     else:
    #         print('Multiprocessing')
    #         # write X1, X2, matrix destination location, num_cores
    #         dist_info = os.path.join(self.D_scratch_dir, 'dist_info.pckl')
    #         with open(dist_info, 'wb') as f:
    #             pickle.dump([X1, X2, matrix_dest_loc, self.multiprocess], f)

    #         # Perform distance calc
    #         os.system(f'python distance_matrix_multiprocess.py {dist_info}')

    #         # Read File
    #         return np.load(matrix_dest_loc)



    def predict(self, X_test):
        print('Property not implemented, use derived class model')
        
    def _is_pos_def(self, K):
        return np.all(np.linalg.eigvals(K) >= 0.)

    def fit(self, X_train, y_train, refit=False):
        print('######### Fitting GPR Model #########')
        start = time.perf_counter()
        dmat_loc = os.path.join(self.D_scratch_dir, 'D_mat.npy')
        if os.path.isfile(dmat_loc):
            print('Reading distance matrix')
            D=np.load(dmat_loc)
            if not D.shape[0]==len(X_train):
                print('Calculating distance matrix to replace old')
                D=self.distance_matrix(X_train, matrix_dest_loc=dmat_loc)
                if self.multiprocess == 1: np.save(dmat_loc,D)
        else:
            print('Calculating distance matrix')
            D = self.distance_matrix(X_train, matrix_dest_loc=dmat_loc)
            if self.multiprocess == 1: np.save(dmat_loc,D)
        
        self.X_train = X_train
        
        print('Constructing kernel')
        K = self.get_kernel(D)
        
        K += np.eye(len(X_train),len(X_train)) * self.sigma_white**2
        self.K_gradient = self._kernel_gradient(K,D)

        # Scale to 0 mean unit variance
        if self.normalize_y:
            self._y_train_mean = np.mean(y_train, axis=0)
            self._y_train_std = np.std(y_train, axis=0)
            y_train = (y_train - self._y_train_mean) / self._y_train_std
            self.y_train = np.array(y_train, ndmin=2).T
        else:
            self.y_train = np.array(y_train, ndmin=2).T
        
        try:
            print('Cholesky decomposition')
            self.L = cholesky(K, lower=True)
        except np.linalg.LinAlgError: 
            print('Linalg error, continuing with last value')
            os.system('echo "Linalg error {}" >> laerrs.txt '.format(len(self.y_train)))
        except Exception as e:
            print(e)         

        print('Cholesky inverse triangular solve')
        self.alpha = cho_solve((self.L, True), self.y_train)
        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        L_inv_T = L_inv.copy().T # Takes up additionally memory, but allows for parallel dot product evaluation
        self.K_inv = L_inv.dot(L_inv_T)
        # del L_inv_T # save some memory # deleted anyways after function is ran

        print(f'######### Finished Fitting GPR Model. Took {(time.perf_counter() - start) / 60.0} mins #########')
               

        
class GPR_base(Kernel_method):
    """ GPR simple implementation following K. Murphy
    X_train has to have shape (n_samples, n_features) 
    """
    
    def predict(self, X_test, return_std=False, return_absolute_std=True):
        print('######### Predicting GPR Model #########')
        start = time.perf_counter()
        d_star_mat_loc = os.path.join(self.D_scratch_dir, 'D_star_mat.npy')
        if os.path.isfile(d_star_mat_loc):
            print('Reading distance matrix')
            D_star=np.load(d_star_mat_loc)
            if D_star.shape[0]!=len(X_test) or D_star.shape[1]!=len(self.X_train):
                print('Calculating distance matrix to replace old')
                D_star = self.distance_matrix(X_test, self.X_train, matrix_dest_loc=d_star_mat_loc)
        else:
            print('Calculating distance matrix')
            D_star = self.distance_matrix(X_test, self.X_train, matrix_dest_loc=d_star_mat_loc)
        if self.multiprocess == 1: np.save(d_star_mat_loc, D_star)       
        
        print('Constructing kernel')
        K_star = self.get_kernel(D_star)
        self.K_star=K_star # rm
        # K_star_star = self.get_kernel(np.array([self.distance_matrix([x],[x])[0][0] for x in X_test])) #+ self.sigma_white**2
        K_star_star = self.get_kernel(self.distance_matrix_pairwise(X_test, X_test)) # Pairwise distances... should of course be all zeros, as above

        #########################################################
        # POTENTIAL ERROR IN CHRISTIANS CODE ABOVE ^ 
        # I believe K_star_star might be wrong here.... it should just be distance_matrix(X_test, X_test) of shape len(X_test) x len(X_test)?
        # With the present code, we underestimate the uncertainty in our predicted measurements, but K_star_star won't impact the model predictions themselves
        #########################################################
        
        print('ystar')
        y_star = np.dot(np.dot(K_star, self.K_inv), self.y_train) 
        if self.normalize_y: y_star = self._y_train_std * y_star + self._y_train_mean # undo normalization
        
        if return_std:
            print('Uncertainty Eval')
            var_y=[]
            for i, k_star in enumerate(K_star):
                var_y.append(K_star_star[i] - np.dot(np.dot(k_star, self.K_inv), k_star.T))
            if self.normalize_y: 
                var_y = np.array(var_y)
                if return_absolute_std: var_y = var_y * self._y_train_std**2 # undo normalization
            
            if np.any(np.isnan(np.squeeze(np.sqrt(var_y)))):
                print('Problem: Nan detected in std')
            
            print(f'######### Finished Predicting GPR Model. Took {(time.perf_counter() - start) / 60.0} mins #########')
            return np.squeeze(y_star), np.squeeze(np.sqrt(var_y))
        
        print(f'######### Finished Predicting GPR Model. Took {(time.perf_counter() - start) / 60.0} mins #########')
        return np.squeeze(y_star)
    
    
    def get_kernel(self, dist):
        print('To be implemented by derived class')


    def _kernel_gradient(self):
        print('To be implemented by derived class')
    
    
    def log_marginal_likelihood(self, eval_gradient=False):
        # print('Log Marginal Likehood determination')
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.y_train, self.alpha)
        log_likelihood_dims -= np.log(np.diag(self.L)).sum()
        log_likelihood_dims -= len(self.X_train) / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
        
        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", self.alpha, self.alpha)  # k: output-dimension
            tmp -= cho_solve((self.L, True), np.eye(len(self.X_train)))[:, :, np.newaxis]
            log_likelihood_gradient_dims = 0.5 * np.einsum("ijl,jik->kl", tmp, self.K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
            return float(log_likelihood), log_likelihood_gradient

        else: 
            return float(log_likelihood)
   

class GPR_tanimoto(GPR_base):
    """ GPR based on minmax Kernel for fingerprint (see explanation in get_tanimoto_distmat) """
    
    def __init__(self,*args, **kwargs):
        if 'metric' in list(kwargs.keys()):
            self.metric = kwargs['metric']
            kwargs.pop('metric')
        else:
            self.metric = 'euclidean'
        
        super(GPR_tanimoto, self).__init__(*args, **kwargs)

        
    # def distance_matrix(self, X1, X2=[]):       
    #     return get_tanimoto_distmat(X1, X2)

    def distance_matrix_pairwise(self, X1, X2=[]):
        X1 = list(X1)
        X2 = list(X2)
        return get_tanimoto_distmat(X1, X2, pairwise=True) 

    def distance_matrix(self, X1, X2=[], matrix_dest_loc=None):
        X1 = list(X1)
        X2 = list(X2)

        if (self.multiprocess == 1) or (len(X1) < 1000):
            # Better to process small vectors with a single core
            return get_tanimoto_distmat(X1, X2)  
        else:
            # Write information
            if matrix_dest_loc is None:
                matrix_dest_loc = os.path.join(self.D_scratch_dir, 'scratch.npy')

            dist_info = os.path.join(self.D_scratch_dir, 'dist_info.pckl')
            with open(dist_info, 'wb') as f:
                pickle.dump([X1, X2, matrix_dest_loc, self.multiprocess], f)

            # Perform distance calc in subprocess and write matrix to 'matrix_dest_loc'
            script_loc = '/home/btpq/bt308495/Thesis/photocatalysis/distance_matrix_multiprocess.py'
            os.system(f'python {script_loc} {dist_info}')
            Dist_mat = np.load(matrix_dest_loc)

            ### Code to remove scratch.npy... somehow

            return Dist_mat   
    
    
    def get_kernel(self, dist):
        return self.constant_kernel**2*(1.-dist)
    
        
    def _kernel_gradient(self,K,D):
        return np.dstack(( (np.full((K.shape[0], K.shape[1]), 2*self.constant_kernel,
                            dtype=np.array(self.constant_kernel).dtype)*K)[:, :, np.newaxis],
                           ( self.constant_kernel**2* np.zeros(K.shape) )[:, :, np.newaxis], 
                           (np.eye(len(self.X_train), len(self.X_train)) * 2*self.sigma_white)[:, :, np.newaxis]
                        ))   
    
    
    
####
# Optimizer
####
from sklearn.utils import check_random_state
    
def _run_gpr_fit_bayesian_opt(X_train, y_train, gprx, starting_values=[1.0, 1., 0.1], 
                              pbounds = {'c':(0.1,3.0),'rbf': (1.,1.),'alpha':(0.001,1.0)}, 
                              niter_local=1, random_state=1):

    ''' Find hyperparameters of GPR model based on  
        log_marginal_likelihood maximization (point estimate) using scipy optimizers '''
    
    random_state = check_random_state(random_state)
    res_vals=[]
    res_dicts=[]
    
    print('Pbounds', pbounds)

    i=0 
    for nit in range(niter_local):

        def log_marginal_likelihood_target_localopt(x, verbose=True):
            nonlocal X_train, y_train, i, gprx
            if verbose: print(f'################## localopt {i} ##################')

            i+=1
            os.system(f'echo "{i} {x[0]} {x[1]} {x[2]}" >> kernel_params_local_opt.txt')
            gprx.set_kernel_params(x[0], x[1], x[2])
            gprx.fit(X_train,y_train)
            log,gradlog=gprx.log_marginal_likelihood(eval_gradient=True)

            if verbose:
                # print(f'******** localopt {i}********')
                print('Model Params:', x)
                print('LL, Gradient LL:', log, gradlog) # Loglikehood, grad LL
            return -log, -gradlog


        if nit!=0:
            starting_values = [random_state.uniform(pbounds['c'][0],pbounds['c'][1]),
                               random_state.uniform(pbounds['rbf'][0],pbounds['rbf'][1]),
                               random_state.uniform(pbounds['alpha'][0],pbounds['alpha'][1])]
        
        print('\n initial guess',starting_values)
        res = minimize(log_marginal_likelihood_target_localopt, 
                   starting_values,
                   method="L-BFGS-B", jac=True, options={'eps':1e-5}, 
                   bounds=[pbounds["c"], pbounds["rbf"], pbounds["alpha"]])
        print('Local (L-BFGS-B) opt {} finished'.format(nit), res['fun'], res['x'])
        res_dicts.append(res)
        res_vals.append(res['fun'])

    res=res_dicts[res_vals.index(min(res_vals))]
    gprx.set_kernel_params(res['x'][0], res['x'][1], res['x'][2])
    print('Best solution after local opt:', res['fun'], res['x'])
    
    gprx.fit(X_train, y_train)    
    
    print(pbounds)
    os.system('echo "pbounds {} {} {}" >> kernel_params.txt'.format('-'.join([str(x) for x in pbounds['c']]),
                                                                       '-'.join([str(x) for x in pbounds['rbf']]),
                                                                       '-'.join([str(x) for x in pbounds['alpha']]) ) )

    return gprx

