import numpy as np
#from TreeExtraction_class import TreeExtraction
#from TreeRepresentation import Jaccard_trees
#from TreeRepresentation import Jaccard_rules
from .TreeRepresentation_utils import tree_splits_to_vector, rule_splits_to_vector


class TreeDissimilarity:

    def __init__(self, clf, set_up, tree_indeces, dissim_method= 'rules',
                 feature_represent= 'weighted', sample=None):
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