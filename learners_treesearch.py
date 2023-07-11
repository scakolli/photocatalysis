import os, time, socket, shutil, random, subprocess
import numpy as np
import pandas as pd
from copy import deepcopy
import multiprocessing as mp

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import osc_discovery.cheminformatics.cheminformatics_mutation as cheminformatics_helpers
from osc_discovery.morphing_typical_OSC_design import operations_typical_osc_design
cheminformatics_helpers.operations=operations_typical_osc_design
cheminformatics_helpers.failed=0
cheminformatics_helpers.stopped=0
cheminformatics_helpers.new_generation=0
cheminformatics_helpers.allow_only_graph_symmetric=True
cheminformatics_helpers.multi_proc = 8
cheminformatics_helpers.preset_structure_check='test_osc_space'
cheminformatics_helpers.mols_last_generation_smiles=[]
cheminformatics_helpers.only_use_brute_force_code=False

from osc_discovery.acquisition_function import get_F, linear_correct_to_b3lyp
from photocatalysis.acquisition_function import F_acqusition
from photocatalysis.gpr_model import GPR_tanimoto, _run_gpr_fit_bayesian_opt
from photocatalysis import visualization_helpers

# print(operations_typical_osc_design.keys())

""" These classes implement the execution of the whole workflow on an HPC system or locally. 

The code can be used to run AML discovery in a testspace with all
molecules and attributes known.  The base_learner class implements all methods 
necessary for basic handling and molecular morphing. It could be used as a base class 
to implement all sorts of workflows. The derived active_learner class implements all 
methods to execute the specific AML discovery task described in the article.
"""




def RW_rank_selection(df, n_sel=100, field='utility_function', random_state=42):
    ''' Roulette Wheel selection, based on ranks '''
    
    smiles_selected=[]
    np.random.seed(random_state)
    df=df.sort_values( by = field, ascending=True)
    df_sel = df.copy()

    if df.shape[0]<=n_sel: return df.molecule_smiles.tolist()

    for i in range(n_sel):
        df_sel = df_sel[~df_sel.molecule_smiles.isin(smiles_selected)]
        ranks =  [(x+1) for x in range(df_sel.shape[0])]
        sum_ranks = np.sum(ranks)
        df_sel['cum_ranks'] = list(np.cumsum(ranks))
        rand_val=np.random.randint(0, sum_ranks)
        smiles_selected.append( df_sel[df_sel.cum_ranks>=rand_val].iloc[0].molecule_smiles )

    return smiles_selected



class base_learner():
    ''' Learner base class. Implements methods (such as queue handling, molecule generation)
    that are common to all learners. '''
    
    def __init__(self,
                 df_initial_population,
                 properties=["XTB1_lamda_h", "ehomo_gfn1_b3lyp"],
                 default_values=[0.,0.],
                 run_mode='queue',
                 jobname_queue='w_orgel',
                 Nbatch_first=100,
                 Nbatch_finish_successful=0,
                 df_reference_chemical_space=[],
                 n_worker_submit=1,
                 two_generations_search=True,
                 worker_submit_file='submit_arthur.sh',
                 submit_command='qsub',
                 queue_status_command='qstat',
                 utility_function_name='utility_function',
                 preset='test_osc_space',
                 dir_save_results='learner_results',
                 dir_scratch='/home/btpq/bt308495/Thesis/scratch',
                 **kwargs):
        ''' Initialze base learner class
            
            Parameters:
            -----------

            df_initial_population: 
                Dataframe containing the initial population
            properties:
                Names of (molecular) properties under scrutiny.
            default_values: 
                Default value of each property
            run_mode:
                Whether to use queue or local submission
            jobname_queue:
                Name used job submission ( in the submit script )
            Nbatch_first:
                Batch selection, number of cases to select for each learning step.
                If the HPC production run is undertaken, this is the number of cases used in the first step.
            Nbatch_finish_successful:
                In the batch selection: How many cases actually need to finish successfully
            df_reference_chemical_space:
                If this reference frame is passed on, calculations are actually not carried out
                but reference values are taken from here
            n_worker_submit:
                How many workers to submit to the queue
            two_generations_search:
                Applying morphing operations twice in consecutive steps to all molecules
            worker_submit_file:
                File that can be submitted to the queue
            submit_command:
                Queue submit command. So far mainly PBS or SLURM are implemented as queue managers
            queue_status_command:
                How can the queue status be queried?
            utility_function_name:
                Utility function field name
            preset:
                Which rules to use to restrict molecular sizes
            dir_save_results:
                Folder in which all results of the run are logged
            dir_scratch:
                Path of scratch directory, for files that are buffered on hard disk, and are not necessary to keep.
        '''
        cheminformatics_helpers.preset_structure_check=preset
        print('Preset: {}'.format(cheminformatics_helpers.preset_structure_check))
        cheminformatics_helpers.mols_last_generation_smiles=[]

        df_population = df_initial_population

        # in the initial frame, generations were set to 1 although it should actually be two, correcting it here
        cheminformatics_helpers.new_generation = df_population.generation.max()

        # Saving all user-settings
        self.list_columns_frame = ['molecule_smiles', 'operation', 'molecule_last_gen', 'generation', 'added_in_round']
        self.df_already_mutated = pd.DataFrame(columns=self.list_columns_frame)
        self.properties = properties
        self.df_population_unique = self._get_unique_population(df_population)
        self.default_values = default_values
        self.worker_submit_file = worker_submit_file
        self.submit_command = submit_command
        self.queue_status_command = queue_status_command
        self.jobname_queue = jobname_queue
        self.ml_rep_field = 'morgan_fp_bitvect'
        self.Nbatch_first = Nbatch_first
        self.n_workers_submit = n_worker_submit
        self.two_generations_search = two_generations_search
        self.utility_function_name = utility_function_name
        self.run_mode = run_mode
        self.dir_save_results = dir_save_results
        self.generation_counter = cheminformatics_helpers.new_generation
        self.added_in_round = cheminformatics_helpers.new_generation
        self.cwd = os.getcwd()
        self.dir_scratch = os.path.join(dir_scratch, dir_save_results)
        self.Nbatch_finish_successful = Nbatch_finish_successful
        self.list_molecules_smiles_mutated=[]
        self.kappa = 0.
        self.preset_structure_check=preset

        print('Settings for batch run: Select every step: {}'.format(self.Nbatch_first))

        if self.Nbatch_finish_successful==0 or not isinstance(df_reference_chemical_space, list):
            self.Nbatch_finish_successful = self.Nbatch_first
            print('Waiting for all cases to be finished')
        else: 
            print('Waiting for {} cases to be finished for each batch'.format(self.Nbatch_finish_successful))
                    
        print('Generation counter:', self.generation_counter)
        
        # Adding a field for the machine-learning representation
        # self.df_population_unique[self.ml_rep_field] = [[]] * self.df_population_unique.shape[0]
        # self.df_population_unique[self.ml_rep_field] = self.df_population_unique[self.ml_rep_field].astype(object)
        self._generate_ml_vectors(self.df_population_unique)
        
        # Removing older worker directory and changing into the cleaned one
        os.system('rm -r {}'.format(dir_save_results))
        os.mkdir(dir_save_results)
        os.chdir(dir_save_results)
        os.system('echo "{}" > hostname.txt'.format(socket.gethostname()))

        # setting up the folder, linking in workers
        # os.system('ln -s $PWD/../../worker_package/* .')
        # os.system(f'ln -s  /home/btpq/bt308495/Thesis/worker/* .')
        print('Now running in {}'.format(os.getcwd()))

        # create dir on scratch for distance matrix, used by gpr model
        if os.path.isdir(self.dir_scratch): 
            shutil.rmtree(self.dir_scratch)
        os.mkdir(self.dir_scratch)
        os.system('ln -s {} scratch_distance_matrix'.format(self.dir_scratch)) 

        # If the user didn't supply an external reference frame, results are obtained through queue calculation
        if isinstance(df_reference_chemical_space, list): 
            df_reference_chemical_space_unique=[]

        else:
            # The user supplied an external reference frame, lets write it to a file, so that all results can
            # be directly read from there instead of performing queue-calculations
            # Note, here we are correcting xTB-predictions to B3LYP level,
            # this is not necessary if actual DFT-results are used.

            for p in self.properties:
                df_reference_chemical_space[p]=linear_correct_to_b3lyp(p, df_reference_chemical_space[p])
            
            # Add utility function
            # y = [df_reference_chemical_space[prop].tolist() for prop in self.properties]
            y = df_reference_chemical_space[self.properties].values.T
            df_reference_chemical_space[self.utility_function_name] = self._get_utility(y)
            df_reference_chemical_space['generation'] = [ int(x) for x in df_reference_chemical_space.generation.tolist() ]
            
            df_reference_chemical_space_unique = self._get_unique_population(df_reference_chemical_space)
                
            # Write results to file results_calculations.txt, where everything is read from
            os.mkdir('molecules_to_calculate_results')
            with open('molecules_to_calculate_results/results_calculations.txt', 'w') as out:
                for i,row in df_reference_chemical_space_unique.iterrows():
                    str_write='mol_{}.smi {}'.format(i, row.molecule_smiles)
                    for p in self.properties: str_write+=' {}'.format(row[p])
                    str_write+='\n'
                    out.write(str_write)
                    
        self.df_reference_chemical_space = df_reference_chemical_space
        self.df_reference_chemical_space_unique = df_reference_chemical_space_unique    
        self.df_population_unique['finished_in_round'] = self.df_population_unique['added_in_round']
        self.df_population_unique.to_json('df_initial.json', orient='split')
       
        # y = [self.df_population_unique[prop].tolist() for prop in self.properties]
        y = self.df_population_unique[self.properties].values.T
        self.df_population_unique[self.utility_function_name] = self._get_utility(y)
        self.df_population_unique.to_json('df_population.json', orient='split')        
        
        print('')
        print('Setup finished')
        print('--------------------------------------') 


    def _get_unique_population(self, df):
        ''' Utility function to get a unique frame with regard of molecules contained '''
        return df.copy().drop_duplicates(subset='molecule_smiles')


    def get_population_completed(self):
        ''' Utility function: Return all completed calculations '''
        return self.df_population_unique[self.df_population_unique.calc_status=='completed']


    def get_population_completed_or_fizzled(self):
        ''' Utility function: Return all completed calculations '''
        return self.df_population_unique[(self.df_population_unique.calc_status=='completed') |
                                         (self.df_population_unique.calc_status=='fizzled')]


    def get_population_not_completed(self):
        ''' Utility function: Return all completed calculations '''
        return self.df_population_unique[(self.df_population_unique.calc_status!='completed') &
                                         (self.df_population_unique.calc_status!='fizzled')]


    def run_calculations_population(self):
        ''' This function checks the current population frame and decides,
        which calculations still need to be carried out. 
        Then they are written to the processing queue and workers are executed.
        '''
        if not os.path.isdir('molecules_to_calculate'):
            os.mkdir('molecules_to_calculate')

        self._update_population_frame()
        
        # just to be sure, but our population should always be unique
        df_population_unique = self._get_unique_population(self.df_population_unique)
        run_count=0
        dict_results = self._get_results()

        for i,row in df_population_unique.iterrows():
            still_to_run=False
            for j, prop in enumerate(self.properties):
                if (row[prop]==self.default_values[j] or np.isnan(row[prop])) \
                    and not row['calc_status']=='fizzled' \
                    and not row['calc_status']=='running' \
                    and not row['calc_status']=='completed' \
                    and not os.path.isfile('molecules_to_calculate/mol_{}.smi'.format(i)):
                    if not row.molecule_smiles in dict_results.keys(): # maybe the calc is already in the results file
                        # Molecule needs to be calculated. Create file
                        with open('molecules_to_calculate/mol_{}.smi'.format(i), 'w') as out:
                            out.write(row.molecule_smiles)
                        
                        # Update calc_status
                        df_sel = self.df_population_unique[self.df_population_unique.molecule_smiles==row.molecule_smiles]
                        for j, row2 in df_sel.iterrows():
                            self.df_population_unique.at[j,'calc_status']='setup'

                        run_count += 1                 
        
        if run_count>0 and not self.check_all_calculations_finished():
            self._submit_worker()


    def _get_results(self):
        ''' Parse the results_calculations.txt file '''
        res_file = 'molecules_to_calculate_results/results_calculations.txt'
        if not os.path.isfile(res_file): return {}
        
        dict_res={}
        with open('molecules_to_calculate_results/results_calculations.txt') as out:
            for line in out.readlines():
                line=line.split()
                smi=line[0]
                props=[float(x) for x in line[1:3]]
                # ADD CODE for processing additional properties in the results file
                # or do so manually outside of learners_treesearch 
                # extra_props = [x for x in line[2:]]
                dict_res[smi]=props
        return dict_res


    def _update_population_frame(self):
        ''' Reads results from the output-directory (molecules_to_calculate_results) and updates the population frame. '''
        
        if not os.path.isdir('molecules_to_calculate_results'):
            # Initial check in the beginning of the run, when no calcs have been done
            print('no results directory, resuming')
            return None    
        
        ### Running/fizzled
        # update frame calc_status
        res_files = [x for x in os.listdir('molecules_to_calculate_results')]
        fizzled = [ int(x.split('.')[0].split('mol_')[-1]) for i,x in enumerate(res_files) if '__fizzled' in x]
        running = [ int(x.split('.')[0].split('mol_')[-1]) for i,x in enumerate(res_files) if '__running' in x]
        
        smi_fizzled = [ open('molecules_to_calculate/mol_{}.smi'.format(idx)).read() for idx in fizzled ]
        smi_running = [ open('molecules_to_calculate/mol_{}.smi'.format(idx)).read() for idx in running ]
        
        for i,fizz_idx in enumerate(fizzled):
            df_sel = self.df_population_unique[self.df_population_unique.molecule_smiles==smi_fizzled[i]]
            if df_sel.shape[0]==0: continue
            if df_sel.calc_status.values[0]=='fizzled': continue
            for j, row in df_sel.iterrows():
                self.df_population_unique.at[j,'calc_status']='fizzled'
                            
        for i,run_idx in enumerate(running):
            df_sel = self.df_population_unique[self.df_population_unique.molecule_smiles==smi_running[i]]
            if df_sel.shape[0]==0: continue
            if df_sel.calc_status.values[0]=='running': continue
            for j, row in df_sel.iterrows():
                self.df_population_unique.at[j, 'calc_status']='running'   

        ### Completed
        # update frame calc_status as well as properties. If no results exist, exit.
        if not os.path.isfile('molecules_to_calculate_results/results_calculations.txt'):
            # No completed mols
            print('no results file, resuming')
            return None
        else:
            # read results file
            dict_results = self._get_results()

        print('Completed before:', self.df_population_unique[self.df_population_unique.calc_status!='completed'].shape[0])
        for i,row in self.df_population_unique[self.df_population_unique.calc_status!='completed'].iterrows():
            if not row.molecule_smiles in dict_results.keys():
                # No results for this mol
                continue
            if self.check_all_calculations_finished(read=False):
                # Update all 'completed' mols
                break

            self.df_population_unique.at[i, 'calc_status']='completed'
            self.df_population_unique.at[i, 'finished_in_round']=self.added_in_round

            for j, p in enumerate(dict_results[row.molecule_smiles]):
                self.df_population_unique.at[i, str(self.properties[j])] = p

        print('Completed after:', self.df_population_unique[self.df_population_unique.calc_status!='completed'].shape[0])
                
        # compute utility
        # y = [self.df_population_unique[prop].tolist() for prop in self.properties]
        y = self.df_population_unique[self.properties].values.T
        self.df_population_unique[self.utility_function_name] = self._get_utility(y)

        return None
        
    
    def check_all_calculations_finished(self, read=True):
        ''' Check which batch calculations are still open '''
        if read:
            self._update_population_frame()
        
        df_finished_population_unique = self.get_population_completed_or_fizzled()
        df_population_unique = self._get_unique_population(self.df_population_unique)

        if df_finished_population_unique.shape[0] != df_population_unique.shape[0]:
             # How many calculations can be left_todo, before proceeding... useful for HPC parallel processing across
             # multiple nodes. If theres only 10 left todo, better not leave all the compute nodes idle while we process
             # these 10... lets already begin fetching the next batch. In our case, with 1 local compute node, this is not
             # necessary, resulting code is redundant and you can basically ignore since threshold = 0 always, when Nbatch_finish_successful = 0

            threshold = self.Nbatch_first - self.Nbatch_finish_successful
            left_todo = df_population_unique.shape[0] - df_finished_population_unique.shape[0]

            if left_todo > threshold:
                if read:
                    print('Finished Systems, Total Systems: {} , {}'.format(df_finished_population_unique.shape[0], df_population_unique.shape[0]))
                    print('Systems to run: {}'.format(left_todo))

                if self.Nbatch_finish_successful!=-1:
                    if read:
                        print('Threshold value to proceed: {}'.format(threshold))
                return False
        
        print('Finished, saving frame to {}'.format(os.path.join(os.getcwd(), 'df_population.json')))
        if read:
            self.df_population_unique.to_json('df_population.json', orient='split')
        return True

    def _submit_worker(self):
        # Workers already linked in previously, simply call them from cwd .../learner_results/
        os.system(f'python {self.worker_submit_file} molecules_to_calculate')
    
    # def _submit_workers_to_cluster(self):
    #     ''' This function checks, whether there are still enough workers running. 
    #     If not, new workers are added to the queueing system.
    #     So far PBS and SLURM are supported
    #     '''
       
    #     print('Worker settings: ')
    #     print('Run mode', self.run_mode)
    #     print(self.queue_status_command)
    #     print(self.submit_command)
    #     print(self.worker_submit_file)

    #     # Get qstat
    #     if self.run_mode=='queue':
    #         if ' ' in self.queue_status_command: list_command = self.queue_status_command.split()
    #         else: list_command = [ self.queue_status_command ]
    #         submit_command = [ self.submit_command, self.worker_submit_file ]
    #     else:
    #         list_command = ['ps','aux','|','grep', self.worker_submit_file ]
            
    #     result = subprocess.run(list_command, stdout=subprocess.PIPE).stdout.decode("utf-8") 
    #     list_queue_jobs_running=[x for x in result.split('\n') if self.jobname_queue in x]
    #     n_workers_submit = self.n_workers_submit - len(list_queue_jobs_running)
    #     if n_workers_submit<=0: 
    #         print('enough workers already')
    #         n_workers_submit=0
    #     else:
    #         print('Detected {} active workers, now submitting {} more'.format(len(list_queue_jobs_running), n_workers_submit))

    #     # If there not enough workers in the queue, submit a new one
    #     for n_w in range(n_workers_submit):
    #         if self.run_mode=='queue': 
    #             result = subprocess.run(submit_command, stdout=subprocess.PIPE)
    #             print('queue submit', submit_command, result)
    #         else: os.system('./{} &'.format(self.worker_submit_file))
    #         time.sleep(10)

    #     # Show the job
    #     result = subprocess.run(list_command, stdout=subprocess.PIPE).stdout.decode("utf-8") 
    #     list_queue_jobs_running=[x for x in result.split('\n') if self.jobname_queue in x]


    def _generate_candidates(self):
        ''' For all molecules that have not been morphed so far, generate offspring by morphing
        All molecules already mutated and those generated by morphing are saved in 
        df_already_mutated, which has the same format as df_population.
        '''
        self.generation_counter+=1        
        
        # Check which ones have to be mutated still
        # Fetch completed smiles
        # get smiles that do not exist in the list_molecules_smiles_mutated
        df_population_unique = self._get_unique_population(self.get_population_completed())
        df_to_morph = df_population_unique[~df_population_unique.molecule_smiles.isin(self.list_molecules_smiles_mutated)]

        self.list_molecules_smiles_mutated += df_to_morph.molecule_smiles.tolist()
        
        print('****************generating candidates*************************')
        print('Applying morphing operations on {} initial candidates. (Gen {})'.\
              format(df_to_morph.shape[0], self.generation_counter))    
        
        if isinstance(self.df_reference_chemical_space, list):
        
            # Run the mutation first time
            new_frames = self._get_new_frames(df_to_morph.molecule_smiles.tolist(), multiproc=True)  
                                
            # Lookahead search mutation on all unique ones
            mutate_list_lookahead = list(set([x.molecule_smiles[0] for x in new_frames]))
            unique_mols_generated_lookahead=[]
            for i in range(self.two_generations_search):
                new_frames_lookahead = self._get_new_frames( mutate_list_lookahead , multiproc=True )
                mutate_list_lookahead = list(set([x.molecule_smiles[0] for x in new_frames_lookahead]))
                new_frames += new_frames_lookahead
                unique_mols_generated_lookahead += list(set([x.molecule_smiles[0] for x in new_frames_lookahead]))
                
            n_unique_mols_generated = len(list(set([x.molecule_smiles[0] for x in new_frames])))
            print('Generated by morphing: {}, Unique ones: {}'.format(len(new_frames), n_unique_mols_generated))
            print('Thereby generated by two-fold morphing: {}'.format(len(set(unique_mols_generated_lookahead))))
            df_candidates = pd.concat(new_frames) 
       

        else: # If we have a precomputed reference frame, we don't need to carry out the morphing steps.
            df_candidates = self.df_reference_chemical_space[self.df_reference_chemical_space.\
                                                             molecule_last_gen.isin(df_to_morph.molecule_smiles)]
            
            mutate_list_lookahead = df_candidates.molecule_smiles.tolist()
            
            for i in range(self.two_generations_search):
                df_candidates_new = self.df_reference_chemical_space[self.df_reference_chemical_space.\
                                                    molecule_last_gen.isin(mutate_list_lookahead)]
                mutate_list_lookahead = df_candidates_new.molecule_smiles.tolist()
                df_candidates = pd.concat( [df_candidates , df_candidates_new] )

            # can get large at larger depths because some might be saved multiple times.
            # Might be useful to remove duplicated entries based on a common identifier. 
            df_candidates = df_candidates[['molecule_smiles','operation',\
                                                   'molecule_last_gen','generation']]

            # added in round will be updated with the value of the run in which it actually entered the population
            print('Found from reference frame (morphing): {}, Unique ones: {}'.format(df_candidates.shape[0], 
                                                               self._get_unique_population(df_candidates).shape[0]))            

        df_candidates = df_candidates[df_candidates.molecule_smiles!=None]
        df_candidates['generation'] = self.generation_counter
        df_candidates['added_in_round'] = self.generation_counter
        self.df_already_mutated = pd.concat([self.df_already_mutated, df_candidates])
 


    def _get_new_frames(self, mols_unique_smi, multiproc=False):
        ''' Helper, pass in a list of smiles and get mutated candidates, either single or multiprocessing. '''
       
        if hasattr(self, 'preset_structure_check'): cheminformatics_helpers.preset_structure_check = self.preset_structure_check
        else: cheminformatics_helpers.preset_structure_check = ''

        if multiproc: # Multiprocessing
            pool = mp.Pool(processes=16)
            new_frames_all = pool.map(cheminformatics_helpers.run_mutation, mols_unique_smi)
            pool.close()
            new_frames_all = [item for sublist in new_frames_all for item in sublist]
        else: # Single processing
            o=-1
            new_frames_all=[]
            for smi in mols_unique_smi:
                o+=1
                if o%10==0 and o>0: print('Mutating {}/{}'.format(o, len(mols_unique_smi)))
                new_frames_all+=cheminformatics_helpers.run_mutation(smi)
        return new_frames_all


    def select_and_add_new_candidates(self):
        ''' This function adds new molecules to the learners population
        For this: New offspring is generated and the selection of new 
        compounds is done based on the specific learner implementation
        '''

        self.added_in_round+=1      

        # Generate candidates from df_population_unique_completed
        self._generate_candidates()

        # Update the candidate list based on the new molecules that entered the population in this step
        df_population_unique_incomplete = self._get_unique_population(self.get_population_not_completed())
        n_select = self.Nbatch_first - df_population_unique_incomplete.shape[0] # Always Nbatch_first in our case, where calculations are completely finished before next run

        # Identify candidates from mutated set, that are not yet in the population
        df_candidates = self.df_already_mutated

        # Candidate logging
        df_candidates.to_json('df_candidates.json', orient='split')
        if not os.path.isdir('log_candidates'): os.mkdir('log_candidates')
        df_candidates.to_json( 'log_candidates/df_candidates_{}.json'.format(self.added_in_round), orient='split' )

        df_candidates_clean = df_candidates[ ~df_candidates.molecule_smiles.isin(self.df_population_unique.molecule_smiles) ]
        df_candidates_unique = self._get_unique_population(df_candidates_clean)

        print('**************selection procedure********************')
        print('Statistics (select_and_add_new_candidates):')
        print('Already mutated size: {}, Unique: {}'.format( self.df_already_mutated.shape[0], 
                                                            self._get_unique_population(self.df_already_mutated).shape[0] ) )
        print('Population size before (Unique): {}'.format( \
                                                    self._get_unique_population(self.df_population_unique).shape[0] ) ) 
        print('Population size before (completed, unique): {}'.format( \
                                                    self._get_unique_population( self.get_population_completed() ).shape[0] ) )         
        print('Candidates: {}, Unique: {}'.format(df_candidates_clean.shape[0], df_candidates_unique.shape[0]))

        # In the AML model, we always fit on completed data
        # Candidates are derived from full population
        df_candidates_unique = self.selection_step(df_candidates_unique, n_select).copy()  
        print('Active learner selected candidates: {}'.format(df_candidates_unique.shape[0]))         

        # Add index and update status
        last_index = np.max(self.df_population_unique.index.tolist())+1
        df_candidates_unique['index']=[x+last_index for x in range(df_candidates_unique.shape[0])]
        df_candidates_unique.set_index('index', inplace=True)
        df_candidates_unique['calc_status']='not written'
        df_candidates_unique['added_in_round']=self.added_in_round

        # Combine with population and call the run script
        self.df_population_unique = pd.concat([self.df_population_unique, df_candidates_unique]) #, sort=True) not sure why sort=True here
        print('Population size after (Unique): {}'.format(\
                                                    self._get_unique_population(self.df_population_unique).shape[0])) 
        print('------------')

        # Run unfinished calculations on the population
        # Note, that population will only be written, when check_all_calculations_finished is called
        # self.run_calculations_population()

        # Might be better to run calculations outside this function...
       

    def selection_step(self):
        """ Placeholder method, this needs to be implemented by the specific learner """
        pass


    def _get_utility(self, y):
        """ Placeholder method, this needs to be implemented by the specific learner """
        return np.array([0.]*len(y))


    def evaluate_performance_external_set(self):
        """ Evaluate the current population against an external frame.
            Placeholder method, this needs to be implemented by the specific learner
        """
        pass












class active_learner(base_learner):
    """ Active machine learning implementation as described in the article, based on infrastructure provided
        by base learner
    """

    def __init__(self,
                 df_initial_population,
                 reduced_search_space=False,
                 Ndeep=500,
                 depth_search=3,
                 kappa=2.5,
                 random_state=42,
                 suffix='',
                 *args, **kwargs):
        
        self.reduced_search_space = reduced_search_space
        self.depth_search = depth_search
        self.Ndeep = Ndeep

        self.MLModels = {}
        self.suffix = suffix

        self.random_state = random_state
        np.random.seed(self.random_state)

        super().__init__(df_initial_population, *args, **kwargs)
        self.kappa = kappa

        self.kernel_parameters={}
        for prop_name in self.properties:
            self.kernel_parameters[prop_name]={'C':1., 'length_scale':1., 'sigma_n':0.1}
       

    
    def _generate_ml_vectors(self, df):
        """ Helper: Add ML-vectors to dataframe """
        
        self.fp_radius = 2 
        fps=[]
        for i,row in df.iterrows():
            fps.append(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(row.molecule_smiles), self.fp_radius))
                        
        df[self.ml_rep_field] = [[]] * df.shape[0]
        df[self.ml_rep_field] = df[self.ml_rep_field].astype(object)
        df[self.ml_rep_field] = fps
        
        return df     
    
    
        
    def _get_ML_model(self, prop_name):
        """ Helper: Fit ML model on one property of the current population frame"""
        
        df_population_unique = self._get_unique_population(self.get_population_completed())
        # df_population_unique = self._generate_ml_vectors(df_population_unique)
        
        print('')
        print('Fitting property: {}'.format(prop_name))
        print('Size of fitting set for ML model (_get_ML_model): {}'.format(df_population_unique.shape[0]))
        
        X_train = df_population_unique[self.ml_rep_field].values
        y_train = df_population_unique[prop_name].values
        
        kernel_params = self.kernel_parameters[prop_name]
        niter_local=5

        gpr = GPR_tanimoto()
        gpr = _run_gpr_fit_bayesian_opt(X_train, y_train, gpr, 
                                     starting_values=[kernel_params['C'], 
                                                      kernel_params['length_scale'],
                                                      kernel_params['sigma_n']], niter_local=niter_local)

        kernel_params['C'] = gpr.constant_kernel
        kernel_params['length_scale'] = gpr.gamma_kernel
        kernel_params['sigma_n'] = gpr.sigma_white      

        self.kernel_parameters[prop_name] = kernel_params
        
        os.system('echo "{} {} {} {} {}" >> kernel_params.txt'.format(kernel_params['C'],
                                                                    kernel_params['length_scale'],
                                                                    kernel_params['sigma_n'],
                                                                    max(df_population_unique['added_in_round'].tolist()),
                                                                    prop_name ))
        
        print('Fitted Kernel c {} rbf {} sigma {} Round: {}'.format(kernel_params['C'],
                                                                kernel_params['length_scale'],
                                                                kernel_params['sigma_n'],
                                                                max(df_population_unique['added_in_round'].tolist()) ))
        return gpr, X_train
        

    
    def _predict_Fi_scores(self, df_candidates_unique, use_previously_fitted_models=False, kappa=-1., verbose=True):
        """ Evaluate Fi score using ML model """
        
        if verbose: print('Size of prediction set for Fi scores (predict_Fi_scores): {}'.format(df_candidates_unique.shape[0]))
        if verbose: print('Removing _mat.npy files', os.getcwd())

        try: os.remove('scratch_distance_matrix/*_mat.npy')
        except: pass
        
        df_candidates_unique = self._generate_ml_vectors(df_candidates_unique)
        # round_carried_out = max(self.get_population_completed()['generation'].tolist())

        # Fit and predict on single properties separately
        y_preds=[]
        stds=[]
        X_test = df_candidates_unique[self.ml_rep_field].values
        for prop_name in self.properties:
            if not use_previously_fitted_models:
                gpr, X_train = self._get_ML_model(prop_name)
            else:
                gpr = self.MLModels[prop_name]; X_train = gpr.X_train

            self.MLModels[prop_name] = deepcopy(gpr)
            y_test_pred, std = gpr.predict(X_test, return_std=True)
            y_preds.append(y_test_pred)
            stds.append(std)
        
        y_preds = np.array(y_preds)
        stds = np.array(stds)
        
        # Scalarize the multiobjective problem
        y_test_pred_comb = self._get_utility(y_preds)
        Fi_scores = self._get_utility(y_preds, std=stds, kappa=kappa, return_array=True)
                
        # Calculate the Fi_score
        df_candidates_unique['Fi_scores']=list(Fi_scores[:,0])
                
        # Remove distance matrix files
        if verbose: print('Removing _mat.npy files', os.getcwd())
        try: os.remove('*_mat.npy')
        except: pass
       
        return df_candidates_unique



    # def _get_utility(self, y, kappa=-1., stds=[], return_array=False):
    #     """ Compute scalarized utility (F and Facq) as described in the article """

    #     utility_values = []
    #     # std_hole_values = []
        
    #     y = np.array(y)
    #     if len(y.shape) == 1: y = np.array(y, ndmin=2).T
    #     stds = np.array(stds)
    #     if len(stds.shape) == 1: stds = np.array(stds, ndmin=2).T
            
    #     if kappa < 0.: kappa = self.kappa

    #     for i in range(y.shape[1]):
    #         if len(stds) == 0:
    #             utility_values.append(get_F(y[:, i], kappa=kappa,
    #                                         return_array=return_array))
    #         else:
    #             utility_values.append(get_F(y[:, i], std=stds[:, i],
    #                                         kappa=kappa, return_array=return_array))
    #             # std_hole_values.append(utility_values[-1][1])

    #     return np.array(utility_values)
    
    def _get_utility(self, y, std=None, kappa=-1., return_array=False):
        ## Adapted for photocatalysis
        if std is None: std = np.array([])
        if kappa < 0: kappa = self.kappa

        return F_acqusition(y, std, kappa, return_array=return_array)

    

    def _evaluate_ml_external_frame(self, df_candidates_unique):
        """ Evaluate ML model against external reference data """

        if isinstance(self.df_reference_chemical_space_unique, list): return None

        # Evaluate combined Fi score externally only if reference frame is available
        X_test = df_candidates_unique[self.ml_rep_field].to_numpy()
        X_test = np.array(df_candidates_unique[self.ml_rep_field].tolist())
        y_preds = []
        stds = []
        round_carried_out = max(self.get_population_completed()['generation'].tolist())

        for prop_name in self.properties:
            
            print("Property: {}".format(prop_name))
            gpr = self.MLModels[prop_name]; X_train = gpr.X_train
            y_test_pred, std = gpr.predict( X_test, return_std=True, return_absolute_std = not 'unscaled_std' in self.suffix)

            df_sel = self.df_reference_chemical_space_unique[self.df_reference_chemical_space_unique.\
                                            molecule_smiles.isin(df_candidates_unique.molecule_smiles)]
            y_test=[]
            for i,row in df_candidates_unique.iterrows():
                df_selx=df_sel[df_sel.molecule_smiles==row.molecule_smiles]
                if df_selx.shape[0]>0: y_test.append(df_selx[prop_name].values[0])
                else: y_test.append(0.); print(row.molecule_smiles)

            if 'lamda_h' in prop_name: 
                visualization_helpers.performance_evaluation(X_test, y_test, y_test_pred, std, X_train, 
                                              name_save='lambda_h', 
                                              round_carried_out=round_carried_out)
            elif 'homo' in prop_name: 
                visualization_helpers.performance_evaluation(X_test, y_test, y_test_pred, std, X_train, 
                                              mode='level', name_save='e_homo',
                                              round_carried_out=round_carried_out)

            y_preds.append(y_test_pred)
            stds.append(std)

        y_preds = np.array(y_preds)
        stds = np.array(stds)

        # Scalarize the multiobjective problem
        Fi_scores = self._get_utility(y_preds, stds=stds, return_array=True)

        # Check combined score        
        df_sel = self.df_reference_chemical_space_unique.copy()\
                 [self.df_reference_chemical_space_unique.molecule_smiles.\
                 isin(df_candidates_unique.molecule_smiles)]            

        y_test_comb = df_sel[self.properties].T
        y_test_comb_utility = self._get_utility(y_test_comb)
        res = self._get_utility(y_test_comb, std=stds, return_array=True)
        #shape: fi_hole, std_hole
        std_hole = Fi_scores[:,1]

        df_sel['y_test_comb'] = y_test_comb
        y_test_comb=[]
        for i,row in df_candidates_unique.iterrows():
            df_selx=df_sel[df_sel.molecule_smiles==row.molecule_smiles]
            if df_selx.shape[0]>0: y_test_comb.append(df_selx['y_test_comb'].values[0])
            else: y_test_comb.append(0.)

        ### DEBUG, NO y_test_pred_comb??
        # visualization_helpers.performance_evaluation(X_test, y_test_comb, y_test_pred_comb, std_hole, X_train, name_save='F', 
        #                                              mode='combined', round_carried_out=round_carried_out)


        
    def evaluate_performance_external_set(self):
        """ Evaluate the current population against an external frame. """

        df_candidates = self.df_already_mutated
        df_candidates[~df_candidates.molecule_smiles.isin(self.df_population_unique.molecule_smiles)]
        res_dict = visualization_helpers.get_results_dictionary(self.get_population_completed(),
                                                                self.df_reference_chemical_space,
                                                                df_candidates)
        return res_dict


    
    def _reduced_search_selection(self, df_candidates_unique, random_state=-1):
        """ This is the method for search space reduction  """

        df_candidates_identified = df_candidates_unique[df_candidates_unique.molecule_smiles=='']
        df_candidates_identified_unique = df_candidates_identified.copy()

        if random_state<0: random_state=self.random_state
        
        # Lets select initial set of molecules
        smi_selected_current_round = RW_rank_selection( self._get_unique_population(self.get_population_completed()),
                                                        n_sel=self.Ndeep, random_state = random_state )
        
        print('###############################################')
        print('Tree selection starting')
        print('Max depth: {}'.format(self.depth_search))
        print('Maximum number of candidates: {}'.format(self.Ndeep))
        np.random.seed(random_state)

        self.ranking_selection='linear'

        Fi_scores={}

        # lets go deeper
        for d in range(self.depth_search):
            
            print( 'Selection round depth {}. Smiles to morph {}'.format(d, len(smi_selected_current_round) ) )

            if d==0: 
                
                df_candidates_identified_current = self.df_already_mutated[ self.df_already_mutated.molecule_last_gen.isin( smi_selected_current_round ) ]
                                
                if len(set(smi_selected_current_round) - set(self.df_already_mutated.molecule_last_gen.tolist())):
                    print('Error, d=0: Not all molecules had already been morphed')
                
                df_candidates_identified_current_unique = df_candidates_identified_current[~df_candidates_identified_current.molecule_smiles.\
                                                          isin(self.df_population_unique.molecule_smiles)].drop_duplicates(subset='molecule_smiles')
                
            else: 
                if not isinstance(self.df_reference_chemical_space, list):
                    df_candidates_identified_current = self.df_reference_chemical_space[ self.df_reference_chemical_space.molecule_last_gen.isin( smi_selected_current_round ) ]
                else:
                    df_candidates_identified_current = pd.concat(self._get_new_frames( smi_selected_current_round , multiproc=True ))

            df_candidates_identified_current_unique = df_candidates_identified_current[~df_candidates_identified_current.molecule_smiles.\
                                                          isin(self.df_population_unique.molecule_smiles)].drop_duplicates(subset='molecule_smiles')

            print('Selection round depth {}. Morphed {} molecules. Candidates identified {}'.format(d, 
                                          len(smi_selected_current_round), df_candidates_identified_current_unique.shape[0] ) )

            if d==0: 
                df_candidates_identified_unique = self._predict_Fi_scores( df_candidates_identified_current_unique )
            else: 
                df_candidates_identified_unique = self._predict_Fi_scores( df_candidates_identified_current_unique, 
                                                                           use_previously_fitted_models=True )
 
            for i,row in df_candidates_identified_unique.iterrows(): Fi_scores[row.molecule_smiles] = row.Fi_scores
            smi_selected_current_round = RW_rank_selection(df_candidates_identified_unique, n_sel=self.Ndeep,
                                                           field='Fi_scores', random_state=random_state)
            df_candidates_identified = pd.concat( [ df_candidates_identified_current,  df_candidates_identified ] )
        

        df_candidates_identified['added_in_round'] = self.generation_counter
        df_candidates_identified_unique = df_candidates_identified[~df_candidates_identified.molecule_smiles.isin(self.df_population_unique.molecule_smiles)]\
                                                    .drop_duplicates(subset='molecule_smiles')

        df_candidates_identified_unique['Fi_scores'] = 0.
        last_index = np.max(self.df_population_unique.index.tolist())+1
        df_candidates_identified_unique['index']=[x+last_index for x in range(df_candidates_identified_unique.shape[0])]
        df_candidates_identified_unique.set_index('index', inplace=True)

        for i,row in df_candidates_identified_unique.iterrows():
            df_candidates_identified_unique.at[i, 'Fi_scores'] = Fi_scores[row.molecule_smiles]

        if not os.path.isdir('log_candidates_rws'): os.mkdir('log_candidates_rws')
        df_candidates_identified_unique.to_json('log_candidates_rws/df_candidates_{}.json'.format(self.generation_counter), orient='split')
 
        print( 'Tree selection finished. Identified candidates: {}'.format(df_candidates_identified_unique.shape[0]) )
        df_candidates_identified_unique = self._generate_ml_vectors(df_candidates_identified_unique)
        
        return df_candidates_identified_unique


                  
    def selection_step(self, df_candidates_unique, n_select):
        """ Active learner selection step: Either select from full candidate list or resort 
            to the reduction of search space. """
        
        try: os.system('rm *_mat.npy')
        except: pass

        if self.reduced_search_space>0:
            
            df_candidates_unique = df_candidates_unique[df_candidates_unique.molecule_smiles=='']
 
            df_candidates_identified_unique = self._reduced_search_selection(df_candidates_unique)
            
        
            o=0
            while df_candidates_identified_unique.shape[0] < n_select:
                
                o+=1; 
                if o>10: break    
                df_candidates_identified_unique = self._reduced_search_selection(df_candidates_unique,
                                                                                 random_state=self.random_state+o)
            
            self._evaluate_ml_external_frame( df_candidates_identified_unique )
            
            df_selected = df_candidates_identified_unique.iloc[np.flip(\
                    np.argsort(df_candidates_identified_unique['Fi_scores'].tolist()),axis=0)[0:n_select]].copy()

            return df_selected


        if df_candidates_unique.shape[0] > self.Nbatch_first:
            print('***************Nbatch Satisfied******************')
            
            df_candidates_unique = self._predict_Fi_scores(df_candidates_unique)
            print('df_candidates_unique', df_candidates_unique.shape)

            self._evaluate_ml_external_frame(df_candidates_unique)
            
            Fi_scores = df_candidates_unique['Fi_scores'].values

            # Pick n_select (Nbatch in our case) largest utility candidates
            df_candidates_unique = df_candidates_unique.iloc[np.flip(np.argsort(Fi_scores),axis=0)[:n_select]]
         
        return df_candidates_unique
    


######## Standalone functions without class reference

def get_unique_population(dframe):
    return dframe.copy().drop_duplicates(subset='molecule_smiles')

def get_population_completed(dframe):
    return dframe.copy()[dframe.calc_status=='completed']

def generate_ml_vectors(dframe, ml_rep_field='morgan_fp_bitvect'):
        """ Helper: Add ML-vectors to dataframe """
        df = dframe.copy()

        fp_radius = 2 
        fps=[]
        for i,row in df.iterrows():
            fps.append(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(row.molecule_smiles), fp_radius))
                        
        df[ml_rep_field] = [[]] * df.shape[0]
        df[ml_rep_field] = df[ml_rep_field].astype(object)
        df[ml_rep_field] = fps
        
        return df   
    
def get_ML_model(dframe, prop_name, ml_rep_field='morgan_fp_bitvect'):
    """ Helper: Fit ML model on one property of the current population frame"""
    df_population_unique = get_unique_population(get_population_completed(dframe))
    df_population_unique = generate_ml_vectors(df_population_unique)
    
    print('')
    print('Fitting property: {}'.format(prop_name))
    print('Size of fitting set for ML model (_get_ML_model): {}'.format(df_population_unique.shape[0]))
    
    X_train = df_population_unique[ml_rep_field].values
    y_train = df_population_unique[prop_name].values
    
    kernel_params = {'C':1., 'length_scale':1., 'sigma_n':0.1}
    niter_local=5

    gpr = GPR_tanimoto()
    gpr = _run_gpr_fit_bayesian_opt(X_train, y_train, gpr, 
                                    starting_values=[kernel_params['C'], 
                                                    kernel_params['length_scale'],
                                                    kernel_params['sigma_n']], niter_local=niter_local)

    kernel_params['C'] = gpr.constant_kernel
    kernel_params['length_scale'] = gpr.gamma_kernel
    kernel_params['sigma_n'] = gpr.sigma_white      
    
    os.system('echo "{} {} {} {} {}" >> kernel_params.txt'.format(kernel_params['C'],
                                                                kernel_params['length_scale'],
                                                                kernel_params['sigma_n'],
                                                                max(df_population_unique['added_in_round'].tolist()),
                                                                prop_name ))
    
    print('Fitted Kernel c {} rbf {} sigma {} Round: {}'.format(kernel_params['C'],
                                                            kernel_params['length_scale'],
                                                            kernel_params['sigma_n'],
                                                            max(df_population_unique['added_in_round'].tolist()) ))
    return gpr, X_train, kernel_params

def get_results():
        ''' Parse the results_calculations.txt file '''
        res_file = 'molecules_to_calculate_results/results_calculations.txt'
        if not os.path.isfile(res_file): return {}
        
        dict_res={}
        with open('molecules_to_calculate_results/results_calculations.txt') as out:
            for line in out.readlines():
                line=line.split()
                smi=line[0]
                props=[float(x) for x in line[1:3]]
                # ADD CODE for processing additional properties in the results file
                # or do so manually outside of learners_treesearch 
                # extra_props = [x for x in line[3:]]
                dict_res[smi]=props
        return dict_res
