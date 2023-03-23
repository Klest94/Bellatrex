from TreeExtraction_class import TreeExtraction
from sklearn.model_selection import ParameterGrid
from utilities import plot_preselected_trees, rule_print_inline
from plot_tree_patch import plot_tree_patched
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
#from utilities import tree, plot_my_tree

import warnings
#warnings.filterwarnings('warning',category=ConvergenceWarning, module='sklearn')  
import numpy as np

class Bellatrex:
    
    FONT_SIZE = 20
    DPI_FIGURE = 300
    
    def __init__(self, clf, set_up, verbose,
                 proj_method="MDS",
                 dissim_method="rules",
                 feature_represent="weighted",
                 p_grid = {"n_trees": [0.2, 0.5, 0.8],
                           "n_dims": [2, 5, None],
                           "n_clusters": [1,2,3],
                             },
                 # n_trees=[0.2, 0.5, 0.8],
                 # n_dims=[2, 5, None],
                 # n_clusters=[1,2,3], 
                 pre_select_trees="L2",
                 fidelity_measure="L2", 
                 n_jobs=1,
                 plot_GUI=False,
                 output_explain=True,
                 plot_max_depth=None,
                 ys_oracle = None):
        
        self.clf = clf #(un)fitted instance of R(S)F
        self.set_up = set_up 
        self.proj_method = proj_method
        self.dissim_method = dissim_method
        self.feature_represent = feature_represent
        self.p_grid = p_grid
        self.pre_select_trees = pre_select_trees
        self.fidelity_measure = fidelity_measure
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.plot_GUI = plot_GUI
        self.plot_max_depth=plot_max_depth
        self.ys_oracle = None

        
    def fit(self, X, y): #works as inteded   #add n_jobs
    
        default_set_keys = set(["n_trees", "n_dims", "n_clusters"])

        if set(self.p_grid.keys()) != set(["n_trees", "n_dims", "n_clusters"]):
            warnings.warn("The hyperparameter list contains unexpected keys," \
                          "other from {}. Ignoring extra parameters".format(default_set_keys))
    
        if "n_trees" not in self.p_grid.keys():
            self.n_trees = [0.2, 0.5, 0.8] # set to default if not existing
        else:
            self.n_trees = self.p_grid["n_trees"]           #CAN BE A LIST

        if "n_dims" not in self.p_grid.keys():
            self.n_dims = [None] # set to default if not existing
        else:
            self.n_dims = self.p_grid["n_dims"]             #CAN BE A LIST
    
        if "n_clusters" not in self.p_grid.keys():
            self.n_clusters = [1, 2, 3] # set to default if not existing
        else:
            self.n_clusters = self.p_grid["n_clusters"]     # CAN BE A LIST    
            
        
        if self.verbose >= 0:
            print("fitting and validating inputs...")
        
        if min(self.n_trees) <= 0:
            raise ValueError("n_trees must be > 0")
        
        if min(self.n_trees) < 1.0 and max(self.n_trees) > 1.0:
            raise ValueError("list of n_trees must be either a proportion \
                             in the (0,1] interval, or the original, integer \
                                     valued number of tree learners.")
                              
        if max(self.n_trees) > self.clf.n_estimators:
            raise ValueError("n_trees cannot be greater than n_estimators")        
        
        #if proportions fo trees are given correctly, as expected
        if 0 < max(self.n_trees) <= 1.0 and 0 < min(self.n_trees):
            # round to closest integer
            self.n_trees = (np.array(self.n_trees)*self.clf.n_estimators+0.5).astype(int)
            
        
        self.clf = self.clf.fit(X, y)
        if self.verbose >= 0:
            print("... fitting complete")
        if self.verbose >= 2:
            print("oracle_sample is: {}".format(self.ys_oracle))
            
        return self
    
    def predict(self, X, idx, as_object=True): 
        
    #def predict_and_show, better? and .predict only for local_w_pred ?
        
        sample = X[idx:idx+1]
        #sample = X.iloc[idx,:]

        if self.ys_oracle != None:
            oracle_sample = self.ys_oracle.iloc[idx] #pick needed one
        else:
            oracle_sample  = None

        
        ####### tuning phase here: consider moving INSIDE TreeExtraction
                
        param_grid = {              #lists or single entries
            "n_trees": self.n_trees, 
            "n_dims" : self.n_dims,
            "n_clusters": self.n_clusters
            }
        
        for key in param_grid:
            if not isinstance(param_grid[key], (list, np.ndarray)):
                param_grid[key] = [param_grid[key]] # single entry to list
        
        
        grid_list = list(ParameterGrid(param_grid))
        best_perf = -np.inf
        
        ETrees = TreeExtraction(self.proj_method, self.dissim_method,
                                self.feature_represent,
                                # referred as (\tau, d, K) in the paper
                                self.n_trees, self.n_dims, self.n_clusters, 
                                self.pre_select_trees,
                                self.fidelity_measure,
                                self.clf,
                                oracle_sample,
                                self.set_up, sample, self.verbose)
        
        
        ''' the TreeExtraction method does most of the computation,
        Hyperparamter optmimisation takes place here, where all possible 
        TreeExtraction( \tau, d, K)  candidates are compared and the hyperparameter
        combination with highest fidelity is selected.
        '''
        
        # setting default "best", params in case of error
        best_params = {"n_clusters": 2, "n_dims": 2, "n_trees": 20}
        
        
        if self.n_jobs > 1:
            warnings.warn('Multiprocessing does not work yet, do not use'
                          'n_jobs > 1 (currently: {}). Setting n_jobs=1 for now'.format(self.n_jobs))
            self.n_jobs=1
        
        if self.n_jobs == 1:
            for params in grid_list: #tuning here:
                try:
                    candidate = ETrees.set_params(**params).extract_trees()
                    perf = candidate.score(self.fidelity_measure, oracle_sample)
                except: # e.g. a ConvergeWarning from kmeans
                    print('Warning, something went wrong, skipping candidate:', params)
                    perf = -np.inf
                    
                if self.verbose >= 5:
                    print("params:", params)
                    print("fidelity current candidate: {:.4f}".format(perf))
    
                if perf > best_perf:
                    best_perf = perf
                    best_params = params

        # this piece of code does not work yet                    
        # elif self.n_jobs > 1:
        #     warnings.warn('Not implemented correctly yet')
            
        #     #function to be called in parallel processing:
        #     def run_candidate(**params):
        #         try: #better replace with real exception or
        #         # add more conditions...
        #             candidate = ETrees.set_params(**params).extract_trees()
        #             perf = candidate.score(self.fidelity_measure, oracle_sample)
        #         except: # e.g. a ConvergeWarning from kmeans
        #             warnings.warning('Warning, something went wrong, skipping candidate:', params)
        #             perf = -np.inf
        #         return perf, params
            
        #     perfs, params_list = zip(*Parallel(n_jobs=self.n_jobs, prefer="threads")(
        #             delayed(run_candidate)(**params) for params in grid_list))
            
        #     best_idx = np.argsort(perfs)[::-1][0] # take top performing index
        #     best_perf = perfs[best_idx]
        #     best_params = params_list[best_idx]
        
            
        if best_perf == -np.inf: # if still the case, everything went wrong...
            warnings.warn("The GridSearch did not find any optimum, setting default parameters")
            

        # closed "GridSearch" loop, storing score of the best configuration
        #if best_perf > -np.inf: # everything alright
        tuned_method = ETrees.set_params(**best_params).extract_trees() # treeExtraction object
        #instant_method = tuned_method.extract_trees() # TreeExtraction object        
        sample_score = tuned_method.score(self.fidelity_measure, oracle_sample)
                        
        final_extract_trees = tuned_method.final_trees_idx
        final_cluster_sizes =  tuned_method.cluster_sizes
        
        if self.verbose >= 1:
            print("best params:", best_params)
            print("Achieved fidelity: {:.4f}".format(best_perf))
            print("(Tuned according to {})".format(self.fidelity_measure))
            
        if self.verbose >= 2: #and sample_info.final_trees_idx > 1
            print("final trees indeces: {}".format(final_extract_trees))
            print("final cluster sizes: {}".format(final_cluster_sizes))
        
        if self.verbose >= 3:

            #if tuned_method.n_trees == 50 and tuned_method.n_clusters >= 2:

            plot_kmeans, plot_data_bunch= tuned_method.preselect_represent_cluster_trees()
            
            plot_preselected_trees(plot_data_bunch, plot_kmeans,
                                   tuned_method, final_extract_trees,
                                   base_font_size=self.FONT_SIZE, 
                                   plot_dpi=self.DPI_FIGURE)
            
            # from utilities import plot_no_clustering
            # plot_no_clustering(plot_data_bunch)  # for a plot in manuscript
            
        if self.verbose >= 4.0 and self.plot_GUI  == False:
            
            for tree_idx, cluster_size in zip(final_extract_trees, final_cluster_sizes):
                
                rule_print_inline(self.clf[tree_idx], sample,
                                  cluster_size/np.sum(final_cluster_sizes))
    
            
        if self.verbose >= 4.0 and self.plot_GUI == True:
            
            from GUI_plots_code import plot_with_interface
            plot_with_interface(plot_data_bunch, plot_kmeans,
                                tuned_method)
            
            try:
                os.remove("colourbar0.png")
                os.remove("colourbar1.png")
            except:
                warnings.warn("\'colourbar*.png\' file not found")
            #except:
            #    print("Tree representation probably collapsed to a d < 2 space")
        
        '''     return format:
            - tuned_method.score(self.fidelity_measure) is a float
            - tuned_method.local_prediction() is an array of shape (#labels,)
                - REMARK: single output local_prediction() is [float]
            - tuned_method is a TreeExtraction object with extra info:
                -(clf, cluster info, optimal hyperparams, set_up, 
                  projection method, dissim. measure...)
        '''
        
        
        tuned_method.sample_score = sample_score

        return tuned_method.local_prediction(), tuned_method # or return just .local_prediction ?
        #sample_score
            
        ###	improve output with
        '''
			Bunch(fidelity=str(self.fidelity_measure),
				score=tuned_method.score(self.fidelity_measure),
				local_pred=tuned_method.local_prediction(),
				n_trees= etc etc)
        '''
        
    def predict_survival_curve(self, X, idx):
        if not self.set_up in ["surv", "survival"]:
            raise ValueError("Input set-up is not a time-to-event!")
        return ValueError("Not implemented yet")
    
    #def predict_median_surv_time()
    
    def predict_proba(self, X, idx):
        self.X = X
        self.idx = idx
        
        if not self.set_up in ["binary", "bin"]:
            raise ValueError("predict_proba not available outside binary setup")
            
        #return self.predict(X, idx, as_object=True): 
        
        y = np.zeros([X.shape[0], self.clf.n_classes_])
        for i in range(X.shape[0]):
            for j in range(self.clf.n_classes_):
                y[i,j] = self.predict(X, i)[j] #other outputs are TreeExtraction method and other         
        return y
            
        
        
        
    