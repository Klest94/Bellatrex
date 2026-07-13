import numpy as np
import pandas as pd
from bellatrex.tree_dissimilarity import TreeDissimilarity
from bellatrex.tree_representation_utils import rule_splits_to_vector
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_avg_dissimilarity_computation():

    vector_1 = np.array([0, 0.2, 0.8, 0])
    vector_2 = np.array([0.3, 0, 0.7, 0])

    expected_dissimilarity = 0.461538461  # ( equal to 1 - 0.7/1.3)

    dissim_class = TreeDissimilarity(
        clf=None, set_up=None, tree_indeces=None
    )  # TODO: Vectorization functions do not need to be class methods

    dissim_matrix = dissim_class.vectors_to_dissim_matrix([vector_1, vector_2])
    avg_dissimilarity = dissim_class.compute_avg_dissimilarity(dissim_matrix)

    assert np.isclose(avg_dissimilarity, expected_dissimilarity, rtol=1e-5)


def test_dissimilarity_matrix_computation():
    vector_1 = np.array([0, 0.2, 0.8, 0])
    vector_2 = np.array([0.3, 0, 0.7, 0])
    vector_3 = np.array([0.5, 0.5, 0, 0])

    # Manually computed dissimilarity matrix elements:
    expected_matrix = np.array(
        [
            [0.0, 0.461538461, 0.888888888],
            [0.461538461, 0.0, 0.82352941],
            [0.888888888, 0.82352941, 0.0],
        ]
    )
    dissim_object = TreeDissimilarity(clf=None, set_up=None, tree_indeces=None)
    dissim_matrix = dissim_object.vectors_to_dissim_matrix([vector_1, vector_2, vector_3])

    assert np.allclose(dissim_matrix, expected_matrix, atol=1e-5)


def test_avg_dissim_and_matrix_function():

    # Generate deterministic data
    X, y = make_classification(n_samples=30, n_features=5, random_state=42)

    # Fit deterministic forest
    clf = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=0)
    clf.fit(X, y)

    test_sample = pd.Series(X[0])
    dissim_object = TreeDissimilarity(
        clf=clf, set_up="binary", tree_indeces=[0, 1], sample=test_sample
    )  # TODO: Vectorization functions do not need to be class methods

    vector_1 = rule_splits_to_vector(clf, 0, feature_represent="weighted", sample=test_sample)
    vector_2 = rule_splits_to_vector(clf, 1, feature_represent="weighted", sample=test_sample)

    dissim_matrix = dissim_object.vectors_to_dissim_matrix([vector_1, vector_2])
    avg_dissimilarity = dissim_object.compute_avg_dissimilarity(dissim_matrix)

    alternative_avg_dissim, alternative_matrix = dissim_object.return_avg_dissim_and_matrix()

    assert np.isclose(avg_dissimilarity, alternative_avg_dissim, rtol=1e-5)
    assert np.allclose(dissim_matrix, alternative_matrix, rtol=1e-5)


# print("Let's get started")
test_avg_dissim_and_matrix_function()
# # test_real_tree_vectors()
print("Test completed successfully.")
