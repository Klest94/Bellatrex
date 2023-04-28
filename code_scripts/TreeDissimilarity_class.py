import numpy as np
#from TreeExtraction_class import TreeExtraction
#from TreeRepresentation import Jaccard_trees
#from TreeRepresentation import Jaccard_rules
from code_scripts.TreeRepresentation_utils import tree_splits_to_vector, rule_splits_to_vector


class TreeDissimilarity:
    
    def __init__(self, clf, set_up, tree_indeces, dissim_method,
                 feature_represent, sample=None):
        self.clf = clf
        self.set_up = set_up
        self.tree_indeces = tree_indeces
        self.dissim_method = dissim_method
        self.feature_represent = feature_represent
        self.sample = sample #is only used for Jaccard_trees dissimilarity
        
    
    def compute_dissimilarity(self):
        
        vectors = self.tree_to_vectors(self.clf, self.tree_indeces,
                                  self.dissim_method, self.feature_represent,
                                  self.sample)
        
        diss_matrix = self.vectors_to_dissim_matrix(vectors)
        avg_dissimilarity = self.distance_matrix_to_float(diss_matrix)

        return avg_dissimilarity, diss_matrix
    
    
    def tree_to_vectors(self, clf, tree_indeces, dissim_method,
                               feature_represent, sample):
        vectors = [] # list of the vector representation ( tree or path)
        
        if tree_indeces is None: # in case no indeces are given, use all the trees of the ensemble
            tree_indeces = range(clf.n_estimators)
                
        if dissim_method == "trees":
            for idx in tree_indeces:
                # weights "by sample size" or "simple" cases are conisidered
                vectors.append(tree_splits_to_vector(clf, idx, feature_represent))
                
                
        if dissim_method == "rules":
            for idx in tree_indeces:
                # weights "by sample size" or "simple" cases are conisidered
                vectors.append(rule_splits_to_vector(clf, idx, feature_represent,
                                                     sample))    
        return vectors
    
    
    def vectors_to_dissim_matrix(self, vector_list):
        
        size = len(vector_list)
        A = np.zeros([size, size])
        
        for i in range(size):
            for j in range(i, size):
                A[i,j] = np.sum(np.minimum(vector_list[i], vector_list[j]))/np.sum(
                                np.maximum(vector_list[i], vector_list[j]))
                A[j,i] = A[i,j]
        
        return 1-A # dissimilarity instead of similarity (A[i,i]= 0)
    
    
    def distance_matrix_to_float(self, dist_matrix):
        # avergaes the OFF_DIAGONAL elements of the matrix
        # if matrix is (1 x 1) return np.nan (division by zero!)
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise Exception("Non square Matrix?")
        if dist_matrix.shape[0] > 1 and dist_matrix.shape[0] > 1:
            return dist_matrix.sum()/dist_matrix.shape[0]/(dist_matrix.shape[1]-1) 
        else:
            return np.nan
        
    # def Jaccard_trees(self, set_up, clf, idx1, idx2): # it's a SIMILARITY measure
    
    #     the_DTree1 = clf.estimators_[idx1].tree_
    #     the_DTree2 = clf.estimators_[idx2].tree_
        
    #     # in leafs the "feature split" is = -2, and leafs must be dropped
    #     fe1 = np.bincount(the_DTree1.feature[the_DTree1.feature > -1 ]) 
    #     fe2 = np.bincount(the_DTree2.feature[the_DTree2.feature > -1 ])
    #     # PROBLEM: some features might be missing (at the tail), fill with zeros
    #     base1 = np.zeros(clf.n_features_)
    #     base2 = np.zeros(clf.n_features_)
    #     #the bincount will drop all features with no splits that come after the last feature with at least one split
    #     # we need ot inlcude them with extra 0-s at the end to preserve vector lenght. We achieve it by doing the following:
    #     base1[:len(fe1)] += fe1
    #     base2[:len(fe2)] += fe2
    
    #     # Jaccardi formula as of Eq.3 in C443 Suppl. material    
    #     #take "elementwise" min and max along indeces and compute ratio
    
    #     return np.sum(np.minimum(base1, base2))/np.sum(np.maximum(base1, base2))
    
        
    # def trees_to_distance_matrix(self, tree_indeces, *args):
    #                                          # *args: dataset, probabs y/n
    #     size = len(tree_indeces)
    #     M = np.zeros([size, size])
    #     for i in range(size):
    #         for j in range(i+1):
    #             if self.dissim_method == "Jaccard":
    #                 M[i,j] = 1-self.Jaccard_index(i, j)
    #                         #1-Jacc for dissimilarity
    #             else:
    #                 raise NameError('input key not recognised, typo?')
    #             M[j,i] = M[i,j]
    #     return M
    
    #def tree_to_vector_representation(self, idx):
        
    # def Jaccard_index(self, idx1, idx2): # it's a SIMILARITY measure
        
    #     # we compute Jaccard similarity index given two tree (indeces)
    #     # of the original R(S)F ensemble learner

    #     the_DTree1 = self.clf.estimators_[idx1].tree_
    #     the_DTree2 = self.clf.estimators_[idx2].tree_
        
    #     # in leafs the "feature split" is = -2, and leafs must be dropped
    #     split_freq1 = np.bincount(the_DTree1.feature[the_DTree1.feature > -1 ]) 
    #     split_freq2 = np.bincount(the_DTree2.feature[the_DTree2.feature > -1 ])
    #     # PROBLEM: some features might be missing (at the tail), fill with zeros
    #     vec1 = np.zeros(self.clf.n_features_)
    #     vec2 = np.zeros(self.clf.n_features_)
    #     #the bincount will drop all features with no splits that come after the last feature with at least one split
    #     # we need ot inlcude them with extra 0-s at the end to preserve vector lenght.
    #     #We achieve it by doing the following:
    #     vec1[:len(split_freq1)] += split_freq1
    #     vec2[:len(split_freq2)] += split_freq2
    
    #     # Jaccard formula as of Eq.3 in C443 Suppl. materil
    #     #take "elementwise" min and max along indeces and compute ratio
    #     return np.sum(np.minimum(vec1, vec2))/np.sum(np.maximum(vec1, vec2))
    #             #Jaccardi sklearn should correspond

    


        