import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(font_scale=2)
sns.set_context("talk")


def get_results_dictionary(df_population, 
                           df_chemical_space, 
                           df_uncomputed=None,
                           plot=True):
    """ Generates dictionary used for logging of runs in testspace """
    
    # Global settings
    threshold = -0.2
    utility_function_field= 'utility_function'
    
    df_population_unique = df_population.drop_duplicates(subset="molecule_smiles")
    df_chemical_space_unique = df_chemical_space.drop_duplicates(subset="molecule_smiles")

    res_dict={}
    res_dict["population_size"] = df_population_unique.shape[0]
    res_dict["population_median"] = df_population_unique[utility_function_field].median()
    res_dict["population_upper_quartile"] = df_population_unique[utility_function_field].quantile(0.75)
    res_dict["chemical_space_median"] = df_chemical_space_unique[utility_function_field].median()
    res_dict["all_samples_above_threshold"] = df_chemical_space_unique[df_chemical_space_unique[utility_function_field] >= threshold].shape[0]
    res_dict["found_samples_above_threshold"] = df_population_unique[df_population_unique[utility_function_field] >= threshold].shape[0]
    res_dict["rel_score"] = float(df_population_unique[df_population_unique[utility_function_field] >= threshold].shape[0]) / \
                            df_population_unique.shape[0]
    print(df_chemical_space_unique[df_chemical_space_unique[utility_function_field]])
    res_dict["tot_score"] = float(df_population_unique[df_population_unique[utility_function_field] >= threshold].shape[0]) / \
                            float(df_chemical_space_unique[df_chemical_space_unique[utility_function_field] >= threshold].shape[0])

    try: 
        df_uncomputed!=None
        df_uncomputed_unique = df_uncomputed.drop_duplicates(subset="molecule_smiles")
        res_dict["uncomputed_size"]=df_uncomputed_unique.shape[0]
    except:
        res_dict["uncomputed_size"]=0
        
    print("Population size {}".format(res_dict["population_size"]))
    print("Population median: {}".format(res_dict["population_median"]))
    print("Population upper quartile: {}".format(res_dict["population_upper_quartile"]))

    print("Chemical space median: {}".format(res_dict["chemical_space_median"]))
    print("All samples above threshold: {}".format(res_dict["all_samples_above_threshold"]))
    print("Found samples above threshold: {}".format(res_dict["found_samples_above_threshold"]))
    print("Rel score: {}".format(res_dict["rel_score"]))
    print("Tot score: {}".format(res_dict["tot_score"]))

    return res_dict



def plot(y_test_pred, y_test, std=[], X_train=[], lim=(0,400), savename='', title=''):
    """ Plot performance of ML model """

    plt.figure(figsize=(5.9,4.7))
    if not len(std): plt.scatter(y_test_pred,y_test,s=4,c=std, alpha=0.15, cmap='seismic')
    else: plt.scatter(y_test_pred,y_test,s=4, alpha=0.3)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.plot((-20,500),(-20,500))
    plt.xticks()
    plt.yticks()
    plt.xlabel('$\mathrm{\lambda_h^{pred}}$ / meV')
    plt.ylabel('$\mathrm{\lambda_h^{true}}$ / meV')
    cbar=plt.colorbar()
    cbar.set_label('$\mathrm{\sigma}$ / meV')
    cbar.ax.tick_params()
    rmsd=np.sqrt(mean_squared_error(y_test,y_test_pred))
    mae=mean_absolute_error(y_test,y_test_pred)
    r2=r2_score(y_test,y_test_pred) 
    print('RMSD',rmsd)
    print('MAE',mae)
    print('R2',r2)
    plt.title('$N_\mathrm{population}$: %d   $N_\mathrm{candidates}$: %d' % (len(X_train),len(y_test_pred)), fontsize=14)
    plt.text(-1.5,-1.95,' $\mathrm{R^2}$: %2.2f \n $\mathrm{MAE}$: %2.2f \n $\mathrm{RMSD}$: %2.2f' % (r2,mae,rmsd), 
         fontsize=14)    
    
    if lim[1]<=0. and lim[1]>-3:
        plt.xlabel('$F_{\mathrm{pred}}$')
        plt.ylabel('$F_{\mathrm{true}}$')
        cbar.set_label('$\sigma$')
    elif lim[1]<-3.:
        plt.xlabel('$\mathrm{E_{HOMO}^{pred}}$ / eV')
        plt.ylabel('$\mathrm{E_{HOMO}^{true}}$ / eV')
        cbar.set_label('$\sigma$ / eV')

    if len(title)>0:
        plt.title(title,pad=12)
    plt.tight_layout()
    
    print('plot',savename, os.getcwd())
    if len(savename)>0 and not 'scratch' in savename: 
        plt.savefig(savename)
    plt.show()
    plt.close()
    

    
def performance_evaluation(X_test,y_test,y_test_pred,std,X_train, 
                          name_save='scratch', mode='lambda', title='', round_carried_out=-1):
    """ Learning result at every learning step, if reference frame is available """
    
    if mode=='lambda':
        plot(y_test_pred, y_test, std, lim=(0,400), X_train=X_train, 
             savename='grafics/fit_gpr_{}_{}.pdf'.format(name_save,round_carried_out))
    elif mode=='combined': 
        plot(y_test_pred, y_test, std, lim=(-2,0), X_train=X_train, 
             savename='grafics/fit_gpr_{}_{}.pdf'.format(name_save, round_carried_out))
    else: 
        plot(y_test_pred, y_test, std, lim=(-7,0), X_train=X_train, 
             savename='grafics/fit_gpr_{}_{}.pdf'.format(name_save,round_carried_out))
    

