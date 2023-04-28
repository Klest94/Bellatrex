import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS #, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import sksurv

from code_scripts.TreeRepresentation_utils import tree_splits_to_vector#tree_vector
from code_scripts.TreeRepresentation_utils import rule_splits_to_vector, add_emergency_noise

#, Jaccard_trees or Jaccard_rules
from sklearn.utils import Bunch
#from sklearn.exceptions import ConvergenceWarning
#from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#from sklearn.tree import export_text

class TreeExtraction:# is it convenient if it inherits?
    
    RAND_SEED = 0
    BINARY_KEYS = ["bin", "binary"]
    SURVIVAL_KEYS = ["surv", "survival"]
    REGRESSION_KEYS = ["regress", "regression", "regr"]
    MTC_KEYS = ["multi-l", "multi-label", "mtc", "multi"]
    MTR_KEYS = ["multi-t", "multi-target", "mtr"]
    
    def __init__(self, proj_method, dissim_method,
                  feature_represent,
                  n_trees, n_dims, n_clusters,
                  pre_select_loss, fidelity_measure,
                  clf, oracle_sample,
                  set_up, sample, verbose,
                  output_explain=True):
        # consider inheriting from LocalMethod, smth like:
        # LocalMethod.__init__(self) (??)
        self.proj_method = proj_method
        self.dissim_method = dissim_method
        self.feature_represent = feature_represent
        self.n_trees = n_trees #repetitive... correct?
        self.n_dims = n_dims
        self.n_clusters = n_clusters
        self.pre_select_loss = pre_select_loss # drop?
        self.fidelity_measure = fidelity_measure
        self.clf = clf
        self.oracle_sample = oracle_sample
        self.set_up = set_up
        self.sample = sample #  X[idx:idx+1]
        self.verbose = verbose # to improve (set levels for example)
        self.final_trees_idx= None #non extracted yet
        self.cluster_sizes = None #non extracted yet
        self.output_explain = output_explain


# class TreeExtraction(Bellatrex):
#     def __init__(self, **kwargs):
#         # If parameters are not provided for TreeExtraction, use the default values from LocalMethod
#         parent_instance = Bellatrex()
#         for key, value in parent_instance .__dict__.items():
#             if key not in kwargs:
#                 kwargs[key] = value

#         # Call the __init__ method of BaseClass using super() and **kwargs
#         super().__init__(**kwargs)    

        
    # adaptation of n_trees to case where the proportion is given instead, is
    # is handled in the .fit of LocalMethod
    
        
    def get_params(self, deep=True):
        return {"dissim_method": self.dissim_method,
                "proj_method": self.proj_method,
                "n_trees": self.n_trees,
                "n_dims": self.n_dims,
                "n_clusters":  self.n_clusters
                }
   	
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
        '''
        the .extract_trees summarises the whole method's procedure,
        the steps are:
            - tree_matrix is build, and tree "local loss"(1-faithfulness) 
            is calculated
            - pre_selection_trees phase selects top \lambda trees ( tunable)
            - dim_reduction performs MDS to reduce dimensionality (tunable)
            - cluster_and-extract performs K-means and selects trees closest
            to each cluster center, also outputs cluster sizes
            - local predict performs the weighted avergae of the extracted
            trees, based on the inputted cluster sizes 
            - finally the clustering performance is evaluated with the .score
            method. Needed for tuning and for reporting.   
        '''
        
    def extract_trees(self):

        '''this is the main function, does the following:
           raw_tree data:
            - compute tree loss, (and sort for pre-filtering)
            - computes vectos_to_matrix,
            - performs dim_reduction step
           extract_final_trees
            - clusters the output tree-representations and select centers ( with weights)
        '''
        ''' HL bunch properties:
            Bunch(proj_data=proj_trees, ---> the tree representation, projected
                  matrix=tree_matrix,   ---> the original tree representation (kept for debug?)
                  index=HL_trees_idx,   ---> indeces of the orginal tree learners ( order is lost ater pre-selection step)
                  RF_pred=RF_pred,      ---> RF original prediction
                  loss=HL_losses)       ---> distance ( loss) to full RF prediction 
        '''
        kmeans, HL_bunch = self.preselect_represent_cluster_trees()
        
        self.final_trees_idx, \
        self.cluster_sizes = self.extract_final_trees(HL_bunch.proj_data,
                                                              HL_bunch.index, kmeans)
        return self
    

    def preselect_represent_cluster_trees(self):
        
        
        ### pre-selection step:
        tree_local_losses, RF_pred = self.calcul_tree_proximity_loss(self.sample) #w.r.t. to sample we want to explain
        
        HL_trees_idx = np.argsort(tree_local_losses)[:self.n_trees] # sort trees losses
        
        HL_preds = [self.tree_prediction(self.clf[i]) for i in HL_trees_idx]
        HL_losses = np.array([tree_local_losses[i] for i in HL_trees_idx])
        
        
        ''' 
        transform trees to vectors and store everything in matrix,
        the output is matrix of the form (n_HL_trees, n_features),
        indeces of the original trees are also stored through the DataFrame
        '''
        
        dict_trees = {} #empty dictionary to fill: {index : tree-vector}
        if self.dissim_method == "trees":
            for idx in HL_trees_idx:
                dict_trees.update({idx : tree_splits_to_vector(self.clf,
                                            idx, self.feature_represent)})
            tree_matrix = pd.DataFrame.from_dict(dict_trees, "index")
        
        elif self.dissim_method == "rules":
            for idx in HL_trees_idx:
                dict_trees.update({idx : rule_splits_to_vector(self.clf,
                                            idx, self.feature_represent, self.sample)})
            tree_matrix = pd.DataFrame.from_dict(dict_trees, "index")
            
            
        '''for PCA, the tree_representation is enough: tree_vector / rule_vector,
            therefore, no pairwise distance is needed '''
        
        if self.proj_method == "PCA":
            pass #no pairwise distance calculation
           
        # for MDS, a transformation to distance square matrix is needed first
        elif self.proj_method == "MDS":
            
            tree_matrix = self.tranform_to_symm_matrix(tree_matrix)
            
            # in case distances all collapsed to 0
            if np.abs(tree_matrix).max() < 1e-6:
                tree_matrix = add_emergency_noise(tree_matrix)
        
        else: 
            raise KeyError("projection method \'{}\' not recognised.".format(self.proj_method))        
            
        '''  at this point, the a tree_matrix has been elaborated. It already
        is in the appropriate format (e.g. symmetric) and ready to pass
        to the dim_reduction method '''
        
        # reduce dimensionality of the tree representations
        # dim_reduction inherits self.n_dims and self.proj_method
        
        # call dim_reduction step ALSO when n_dims=None. In this case,
        # the method will handle the case appropriately
        # dim_reduction calls either PCA or MDS, according to proJ-method variable
        proj_trees = self.dim_reduction(tree_matrix) # from pandas to numpy. Indexes stored in Bunch
        
        # y needs to contain all floats...
        if not all(isinstance(y, float) for y in HL_preds): 
            if HL_preds[0].size < 2:    # but is single target output
                HL_preds = [float(y) for y in HL_preds] # fix to float
                RF_pred = float(RF_pred)            # fix to float
                
        if not isinstance(RF_pred, float) and RF_pred.size == 1:
            RF_pred = float(RF_pred)            # fix to float
    
        '''  clustering step (indeces retained in Bunch object) '''
        
        HL_bunch = Bunch(proj_data=proj_trees, matrix=tree_matrix, 
                         index=HL_trees_idx,
                         loss=HL_losses,
                         pred=HL_preds,
                         RF_pred=RF_pred,
                         set_up=self.set_up)
        # numpy 2d-array (lambda,2), (lamda, lambda), orig. indeces, losses
        
        #with warnings.catch_warnings():
        #    warnings.filterwarnings("error", category=UserWarning, module='sklearn')
            
        if self.n_clusters > 1: # K-Means is performed on the projected trees
            kmeans = KMeans(n_clusters = self.n_clusters, max_iter=5000,
                    random_state=TreeExtraction.RAND_SEED).fit(proj_trees)
            
        else: #avoid running K-Means (raises warning)
            #output info in the same format so that the cases are indistinguishable
            one_label = np.zeros(proj_trees.shape[0], dtype=np.int8)
            one_center = np.mean(proj_trees, axis=0) 
            # mimic a kmeans instance
            kmeans = Bunch(labels_=one_label, #kmeans.lables_ is a 1D array of 1s only (single class)
                           cluster_centers_= np.array([one_center]) # kmeans.centers_ is a 2D array
                           ) 
                
        return kmeans, HL_bunch
    

    def calcul_tree_proximity_loss(self, sample):
        
        # get number of output labels (only for binary vs multi-label) # IMPROVE, this is a bit werid isn't it?
        #if isinstance(self.clf, sklearn.ensemble.RandomForestClassifier):
        
        if self.clf.n_outputs_ > 1: #multi-output set-up
            tree_preds = np.zeros([self.clf.n_estimators, self.clf.n_outputs_])
        else: #single output set-up (binary, survival, regression)
            tree_preds = np.zeros(self.clf.n_estimators)
               
        my_pre_select_loss =  np.zeros(self.clf.n_estimators)
        
        '''  WARNING:
            In sklearn v.1.1.3 clf[k] does not inherit feature_names_in_ 
            from its parent clf ( trained tree ensemble on X_train a DataFrame)
            
            for this reason, whenever clf[k].predict(sample) is called,
            we call it on the sample.values instead
        '''
        
        
        if self.set_up.lower() in self.BINARY_KEYS:
            RF_pred = self.clf.predict_proba(sample)[0,1]
            #tree_preds = tree_preds.ravel()
            for k in range(self.clf.n_estimators):
                tree_preds[k] = self.clf[k].predict_proba(sample.values)[0,1]
                
        elif self.set_up.lower() in self.SURVIVAL_KEYS + self.REGRESSION_KEYS:
            RF_pred = self.clf.predict(sample)

            for k in range(self.clf.n_estimators):
                tree_preds[k] = self.clf[k].predict(sample.values)
        
        elif self.set_up.lower() in self.MTC_KEYS:
            RF_pred = self.clf.predict_proba(sample) #list of 2d arrays
            #RF_label_list = [lab[:,1][0] for lab in RF_pred]
            RF_label_list = [lab[0][1] for lab in RF_pred] #get prob of posit label
            RF_pred = np.array(RF_label_list).ravel()
            
            # multi-label needs extra adjustments
            for k in range(self.clf.n_estimators): # add column selection
                label_probs_list = self.clf[k].predict_proba(sample.values)
                #label_list = [lab[:,1][0] for lab in label_probs_list]
                label_list = [lab[0][1] for lab in label_probs_list]
                tree_preds[k,:] = np.array(label_list).ravel()
                # predicted (prob.) labels for the given sample
        
        elif self.set_up.lower() in self.MTR_KEYS:
            RF_pred = self.clf.predict(sample) #list of 2d arrays
            #RF_label_list = [lab[:,1][0] for lab in RF_pred]
            #RF_label_list = [lab for lab in RF_pred] #get prob of posit label
            RF_pred = RF_pred.ravel()
            
            for k in range(self.clf.n_estimators): # TODO future: add column selection (here?)
                #tree_preds = self.clf[k].predict(sample).ravel()
                #label_list = [lab[0] for lab in label_probs_list]
                tree_preds[k,:] = self.clf[k].predict(sample.values)
                # predicted (prob.) labels for the given sample
        else:
            raise KeyError("Set-up '{}' not recognized. Check the set_up attribute")
            

        # tree loss (euclidean norm of the predition vector vs real target(s))
        if self.pre_select_loss in ["fidel-L2", "L2"]:
            for k in range(self.clf.n_estimators):
                my_pre_select_loss[k] = np.linalg.norm(RF_pred-tree_preds[k])
        elif self.pre_select_loss in ["fidel-cosine", "cosine", "cos"]:
            my_pre_select_loss[k] = cosine_distances(RF_pred, tree_preds[k])
        else:
            raise KeyError("Pre-selection criterion '{}' not recognized")

        return my_pre_select_loss, RF_pred


    def tree_prediction(self, clf_tree): #(self, HL_indeces, sample):
        
        # same as before: clf_tree does not inherit feature_names_in_ correctly
                
        if hasattr(clf_tree, "predict_proba"):
            if clf_tree.n_outputs_ > 1: # multi-label-classification case
                single_tree_pred = np.array(clf_tree.predict_proba(self.sample.values))[:,:,1]
            else: #binary case
                single_tree_pred = clf_tree.predict_proba(self.sample.values)[:,1][0]
        elif hasattr(clf_tree, "predict"): #regression, survival, multi-target regression
            if isinstance(clf_tree, sksurv.tree.SurvivalTree) or clf_tree.n_outputs_ == 1: # survival OR regression
                single_tree_pred = float(clf_tree.predict(self.sample.values))

            else: #multi-target regression
                # nothing is accurate here, just random
                single_tree_pred = clf_tree.predict(self.sample.values)
        
        else:
            raise ValueError("No case was found for clf_tree, check your input")
                

        return single_tree_pred


    def jaccard_pair_distance(self, tree_1, tree_2):
        return  np.sum(np.minimum(tree_1, tree_1))/np.sum(np.maximum(tree_1, tree_1))
    
    
    def tranform_to_symm_matrix(self, matrix_df_input):
        # assumes the input is a DatFrame (the DataFrame is create just before this 
        # function is called, so I wouldn't even bother checking for type
        matrix_input = matrix_df_input.values # to np.ndarray
        
        #assumes each ROW corresponds to a representation of a tree
        size = matrix_input.shape[0]
        A = np.zeros([size, size])   #this part can be optmisied (and double checked)

        for i in range(size):
            for j in range(i+1):
                    A[i,j] = 1-self.jaccard_pair_distance(matrix_input[i],
                                                     matrix_input[j]) #1-Jacc for dissimilarity
                    A[j,i] = A[i,j]
        return A # no need to store DataFrame, the original indeces
        # are stored in some HL_indeces and in the soon to appear HL_bunch object
        #return pd.DataFrame(A, columns=matrix_df_input.columns,
        #                    index=matrix_df_input.columns)
    
        
    def pre_selection_trees(self, tree_matrix):
        tree_prox_loss = tree_matrix.loss_loss
        HL_trees_idx = np.argsort(tree_prox_loss)[:self.n_trees] # first sort all trees
        
        HL_tree_losses = np.sort(tree_prox_loss)[:self.n_trees] # get LOWEST loss
        HL_trees = tree_matrix.matrix.reindex(index=HL_trees_idx).reindex(columns= HL_trees_idx)
        HL_bunch = Bunch(trees=HL_trees, loss=HL_tree_losses)
        return HL_bunch
    
    
    def dim_reduction(self, HL_trees):
        
        # setting up "tru"number of output dimensions, furtrmore
        # prevent errors from raising if n_dims > n_samples
        if self.n_dims == None:
            true_dims = min(HL_trees.shape[0], HL_trees.shape[1]) #keep all dimensions
        elif self.n_dims >= 1: # should not be needed, but sometimes it is...
            true_dims = min(self.n_dims, HL_trees.shape[0], HL_trees.shape[1])
        else:
            true_dims = self.n_dims # follows sklearn's implementation (amount of explained variance) 
        
        # apply the dim reduciton method from the candidate ETrees object
        if self.proj_method == "PCA" and self.n_dims != None: # if None, skip completely
            m = PCA(n_components=true_dims,
                    random_state=TreeExtraction.RAND_SEED)
            proj_trees = m.fit_transform(HL_trees)

        elif self.proj_method == "MDS":
            m = MDS(n_components=true_dims, metric=True, max_iter=1000, # metric=True, sure?
                    random_state=TreeExtraction.RAND_SEED, dissimilarity='precomputed')
            proj_trees = m.fit_transform(HL_trees)

        else: # method == PCA and n_dims == None: do nothing :-P
            proj_trees = HL_trees
        
        return proj_trees
    
    def extract_final_trees(self, proj_trees_data, origin_indeces, kmeans):
        

        cluster_sizes = np.bincount(kmeans.labels_)
        search_tree = KDTree(proj_trees_data)        
        final_trees = search_tree.query(kmeans.cluster_centers_, k=1,
                                         return_distance=False).flatten()
        # flattening is necessay, otherwise 2D array is outputted
        #search_tree.query returns: (dist, indeces loc), we are interested in indeces only
        # however, pandas indeces are now lost, we need to retrieve them:
        #self.final_trees_idx= proj_trees.index[final_trees].values
        self.final_trees_idx = [origin_indeces[k] for k in final_trees]
            #TreeExtraction.plot_my_clusters(self, proj_trees, kmeans)
        return self.final_trees_idx, cluster_sizes #it's an array, right?
    

    def local_prediction(self): #self clf, X, sample, 
        
        N_trees = np.sum(self.cluster_sizes)
        assert np.shape(self.cluster_sizes)[0] == np.shape(self.final_trees_idx)[0]
        
        if hasattr(self.clf, "predict_proba"):
            preds_list = np.sum([np.multiply(size, self.clf[t].predict_proba(self.sample.values)) 
                      for size, t in zip(self.cluster_sizes, self.final_trees_idx)], 
                                axis=0) # \sum of pred(t)*size
            if self.set_up.lower() in self.MTC_KEYS:
                my_prediction = np.transpose([pred[:, 1][0] for pred in preds_list]) # (n_labels,)
            elif self.set_up.lower() in self.BINARY_KEYS:
                my_prediction = np.transpose([pred[1] for pred in preds_list]) # (1,)
            else:
                raise ValueError("set-up for predict_proba must be binary or multi-label")

        elif hasattr(self.clf, "predict"):
            #survival, regression or MTR set-up (to be renewed?)
            my_prediction = 0.0
            for size, t in zip(self.cluster_sizes, self.final_trees_idx):
                my_prediction += self.clf[t].predict(self.sample.values)*size
            if self.set_up.lower() in self.MTR_KEYS:
                my_prediction = my_prediction.ravel() # not good practice, but it works
            
        else:
            raise ValueError("oracle doesn't have .predict nor .predict_proba method")
            
        # post processig output format here:
        if self.set_up.lower() in self.BINARY_KEYS + self.REGRESSION_KEYS + self.SURVIVAL_KEYS: # binary, regression, survival
            my_prediction = float(my_prediction)
        
        # after summing all predictions, divide by number of estimators:
        return (1/N_trees)*my_prediction
        
    
    ### HERE ADD ORACLE PREDICTION IN THE FUTURE (decouple from RF)
    def oracle_prediction(self): # add, inherit prediction method
        if hasattr(self.clf, "predict_proba"): #and self.set_up.lower() in binary_key_list: , "multi-label", "multi-l"
            raw_output = self.clf.predict_proba(self.sample) #list of probs per label
            if self.set_up.lower() in self.BINARY_KEYS:
                # return size (1,) array, consistent with y_pred and multi-label setup
                return raw_output[0][1] # size () (float) ok?
            
            elif self.set_up.lower() in self.MTC_KEYS:
                y_pred_original = np.array([lab_y[0][1] for lab_y in raw_output])
                return y_pred_original
            
        elif hasattr(self.clf, "predict"):# and self.set_up.lower() in ["surv", "survival"] + [reg, regress, regression]:
            return self.clf.predict(self.sample)
        
        else:
            raise ValueError("prediction attribute does not exist, or wrong \
                              combination is given: \n {} and {}".format(self.clf, 
                                                                  self.set_up))
    
    def score(self, fidelity_measure, oracle_sample):
        y_local = self.local_prediction()
        y_pred = self.oracle_prediction() if oracle_sample is None else oracle_sample  # not always same format as y_local
        
        # y_local and y_pred might not have the exact same format,
        # but the scoring measures still work (as for now)
        
        if fidelity_measure in ["fidel-L2", "L2"]:
            return 1-np.linalg.norm(y_pred-y_local) # 1 - L2 distance
        elif fidelity_measure in ["fidel-cosine", "cosine", "cos"]:
            #it outputs a matrix of distances, we are interested in a 1x1 comparison
            return 1-float(cosine_distances(y_pred.reshape(1,-1), y_local.reshape(1,-1)))
        else:
            raise KeyError("Input {} is not recognised".format(self.fidelity_measure))
            
        
    ''' this code is of poor quality, seldom used '''
    # TODO clean up from this point onwards
    def print_rules(self, save_name_folder, i, y_true=np.nan):
        
        # the name of the folder will be given as an input to the class method
        
        print("Calling print_rules now")

        rf = self.clf
        sample = self.sample #pd Series
        feature_names = list(self.sample.columns)
        weights_dict = dict(zip(self.final_trees_idx, 
                                self.cluster_sizes/np.sum(self.cluster_sizes)))
        
        sample_array = self.sample.to_numpy().reshape(1,-1).astype("float32")
        
        sample_string = "Sample" + str(i) + "_input.txt"
        full_save_sample = os.path.join(save_name_folder, "path-print", sample_string)
        
        dir_save_tree = os.path.join(save_name_folder, "path-print")
        dir_plot_tree = os.path.join(save_name_folder, "plot-save")
        
        directories = [dir_save_tree, dir_plot_tree]
        for new_dir in directories:
            if not os.path.exists(new_dir): #if the data folder does not exists (e.g. new dataset being used)
                os.makedirs(new_dir) # then create the new folder

        '''plotting covariate values of the sample to be explained '''
        with open(full_save_sample, 'w+') as f:
            f.write("###### SAMPLE to explain ######\n")
            for feat in feature_names:
                sample_value = sample[feat].values[0]
                f.write("{:13}: {:7} \n".format(feat, str(sample_value)))
                
        for tree_idx in self.final_trees_idx:
            
            tree_path_string = "Sample" + str(i) + "_Tree" + str(tree_idx) + "_Tree.txt"
            final_leaf_string = "Sample" + str(i) + "_Tree" + str(tree_idx) + "_Leaf.txt"
            final_plot_string = "Sample" + str(i) + "_Tree" + str(tree_idx) + "_Plot.png"
            
            full_save_tree = os.path.join(dir_save_tree, tree_path_string)
            full_save_leaf = os.path.join(dir_save_tree, final_leaf_string)
            full_save_plot = os.path.join(dir_plot_tree, final_plot_string)
            
            DT_path = self.clf[tree_idx].tree_.decision_path(sample_array).toarray().flatten()            
                        
            #if isinstance(rf[tree_idx], sklearn.tree._classes.DecisionTreeClassifier):
            if self.set_up.lower() in self.BINARY_KEYS:
                leaf_print = float(rf[tree_idx].predict_proba(sample)[:,1])
                RF_pred = float(rf.predict_proba(sample)[:,1])
                RF_preds = [float(rf[i].predict_proba(sample)[:,1])
                            for i in range(rf.n_estimators)]
            elif self.set_up.lower() in self.MTC_KEYS:
                # not sure the formatting is OK (probably not)
                leaf_print = rf[tree_idx].predict_proba(sample)[:,1].ravel()
                RF_pred = rf.predict_proba(sample)[:,1].ravel()

            elif self.set_up.lower() in self.REGRESSION_KEYS:
            #elif isinstance(rf[tree_idx], sklearn.tree._classes.DecisionTreeRegressor):
                leaf_print = rf[tree_idx].predict(sample).ravel()
                RF_pred = float(rf.predict(sample)[:,1])
                RF_preds = [float(rf[i].predict(sample)[:,1])
                            for i in range(rf.n_estimators)]

            elif self.set_up.lower() in self.SURVIVAL_KEYS: # is survival case
                leaf_print = rf[tree_idx].predict(sample).ravel() # risk score
                leaf_surv_curve = rf[tree_idx].predict_survival_function([sample], return_array=False)
                
                RF_pred = rf.predict(sample).ravel()
                RF_preds = [rf[i].predict(sample)[:,1].ravel()
                            for i in range(rf.n_estimators)]

                RF_pred_curve = rf.predict_survival_function([sample], return_array=False)
                RF_pred_curves = [rf[i].predict_survival_function([sample], return_array=False)
                                  for i in range(rf.n_estimators)]
                
            else:
                raise KeyError("Set-up {} is not recognised".format(self.set_up))

                
            tree_structure = rf[tree_idx].tree_
                        
            #feature_names = [weird list comprehension including _tree.TREE_UNDEFINED]
                
            intervals = {feat : [-np.inf, np.inf] for feat in feature_names}
            
            # this is weird, why does sample change type all of the sudden?
            if isinstance(sample, pd.Series):
                sample = sample.to_numpy().reshape(-1,1).flatten() #from single column to single line
            elif isinstance(sample, np.ndarray):
                sample = sample.reshape(-1,1).flatten()
            
            if self.output_explain == False:
                print("output_explain is \'False\', no output files are generated")
                
            def current_pred(tree_struct, node_id, indent, set_up, opened_f):
                if set_up.lower() in ["bin", "binary"]:
                    pred_values = tree_struct.value[node_id][0,:]
                    avg_pred = pred_values[1]/np.sum(pred_values)
                    if node_id == 0:
                        opened_f.write("base pred:{} {:#4g}\n".format(indent, avg_pred))
                    else:
                        opened_f.write("curr pred:{} {:#4g}\n".format(indent, avg_pred))

                else:
                    raise Warning("Current node prediction is not yet implemented for this scenario")
                
            
            if self.output_explain == True:
                
                sample_np = sample.values.T.ravel()

                
                with open(full_save_tree, 'w+') as f:
                    
                    ''' print decision path and store intervals that
                        identify the final leaf '''
                    def recurse(node, depth, sample_np, intervals):   
                        
                        
                        indent = "   " * depth
                        if tree_structure.feature[node] != -2: # is not a leaf #_tree.TREE_UNDEFINED:
                            name = feature_names[tree_structure.feature[node]]
                            threshold = tree_structure.threshold[node]

                            if DT_path[node] == 1 and sample_np[tree_structure.feature[node]] <= threshold:
                                intervals[name][1] = threshold # reduce feature upper bound
                                # avoid priting this repreatedly  during recursion
                                DT_path[node] = 0
                                if self.set_up.lower() in ["bin", "binary"]:
                                    current_pred(tree_structure, node, indent, self.set_up, f)
                                f.write("node.{:4}: {}if {:8} <= {:#4g}\n".format(node, indent, name, threshold))
                                # works only for binary case


                            recurse(tree_structure.children_left[node], depth + 1, sample_np, intervals)
                            
                            if DT_path[node] == 1 and sample_np[tree_structure.feature[node]] > threshold:
                                intervals[name][0] = threshold # increase feature lower bound 
                                DT_path[node] = 0
                                if self.set_up.lower() in ["bin", "binary"]:
                                    current_pred(tree_structure, node, indent, self.set_up, f)
                                f.write("node.{:4}: {}if {} > {:#4g}\n".format(node, indent, name, threshold))
                                #print("node.{:4}: {}if {:8} <= {:5}\n".format(node, indent, name, threshold))
                            recurse(tree_structure.children_right[node], depth + 1, sample_np, intervals)
                        else: #it is undefined, it is therefore a leaf (?)
                            if DT_path[node] == 1:
                                # no need to print twice with current_pred function
                                f.write("leaf.{:4}: {}return {:#4g}\n".format(node, indent, leaf_print))
                                
                                #print("predicted:{}\n".format(leaf_print))
                                f.write("###################\n")
                                f.write("Tree weight:   {:.2f}\n".format(weights_dict[tree_idx]))
                                f.write("Tree predicts: {:.4f}\n".format(leaf_print))
                                f.write("RF prediction: {:.4f}\n".format(RF_pred))
                                f.write("True label: {}".format(y_true))
                                
            
                    recurse(0, 1, sample_np, intervals) # problme arises here
                
                    if self.set_up.lower() in ["survival", "surv"]:
                        plt.figure()
                        #plt.title("Tree {} predicting sample {}".format()) not available at this level, it's in main!!
                        plt.plot(leaf_surv_curve[0].x, leaf_surv_curve[0].y)
                        plt.ylim(ymin=0, ymax=1)
                        plt.savefig(full_save_plot)
                        plt.legend()
                        plt.show()
                                
                    if self.verbose >= 5.0:
                        pd.Series(RF_preds).plot.density(color='blue', label="RF predictions")
                        plt.title('RF prediction distribution')
                        plt.axvline(x=RF_pred, label="y pred.")
                        #if self.set_up.lower() in ["regress", "regr", "regression"]:
                        plt.axvline(x=y_true, label="y true", color="green")
                        plt.xlim(0,1)
                        plt.legend()
                        #plt.savefig(full_save_plot)
                        plt.show()
                
                                
                with open(full_save_leaf, 'w') as f:
                    f.write("###### considered covariates ######\n")
                    for j, k in zip(feature_names, range(len(feature_names))):
                        if intervals[j][0] > -np.inf or intervals[j][1] < np.inf: 
                            f.write("{:10}: {:#4g} \n".format(str(j), sample_np[k]))
                    f.write("#### True label: {} ####".format(y_true))

                        
                    f.write("\n###### final intervals ########\n")
            
                    for item in intervals:
                        if intervals[item][0] > -np.inf or intervals[item][1] < np.inf: 
                            f.write("{:#4g} < {:16} <= {:#4g} \n".format(intervals[item][0],
                                                str(item).center(16), intervals[item][1]))
                                            
                    with open(full_save_tree) as f: #printing tree-rule structure on console
                        print(f.read())
                                    
                    with open(full_save_leaf) as f:
                        print(f.read()) #printing (simplified) leaf structure on console
                    
    #return tree_rules, leaf_struct

            
            
            
            

        
        
            
        
        
