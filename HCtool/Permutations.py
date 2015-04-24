# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>,
#         Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause
# Here modified By Dmitry Smirnov

from __future__ import print_function
from __future__ import division

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Parallel, delayed

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state
from sklearn.metrics.scorer import check_scoring
from sklearn.cross_validation import _check_cv

plt.ioff()

def GetPermutations(cls,data_test,labels,crossval,data_train=None,
                    scoring=None,n_permutations=1000,n_jobs=-1,
                    **kwargs):
    ''' Wrapper for permutations '''
    null_cv_scores = permutation_test_score(estimator = cls,
                                            X = data_test,
                                            data_train = data_train,
                                            y = labels.get('regressor'),
                                            cv = crossval,
                                            scoring = scoring,
                                            n_permutations = n_permutations,
                                            n_jobs = -1)           
    return null_cv_scores
    
        
def plot_permutation(null_cv_scores):
        ''' Make hist of permutation scores, with vertical line for max permutation accuracy and mean accuracy ''' 
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        plt.title("Permutation threshold, pval = {0:.3f}".format(null_cv_scores[2]))
        bins = ax.hist(null_cv_scores[1],bins = 20, histtype="stepfilled", color = "#348ABD", alpha = 0.60, normed = True );
        plt.vlines(null_cv_scores[0], 0, max(bins[0])+1, linestyle = "--", linewidth = 2, color = 'red' , label = "Accuracy, {0:.2f}%".format(null_cv_scores[0]))
        plt.vlines(np.max(null_cv_scores[1]), 0, max(bins[0])+1, linestyle = "--", linewidth = 2, label = "Permutation threshold, {0:.2f}".format(np.max(null_cv_scores[1])))
        plt.legend(loc ="upper right")
        
        return fig
        
        
def permutation_test_score(estimator, X, y, data_train=None, cv=None,
                           n_permutations=100, n_jobs=1, labels=None,
                           random_state=0, verbose=0, scoring=None):
    """Evaluate the significance of a cross-validated score with permutations

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like
        The target variable to try to predict in the case of
        supervised learning.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : integer or cross-validation generator, optional
        If an integer is passed, it is the number of fold (default 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.

    n_permutations : integer, optional
        Number of times to permute ``y``.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    labels : array-like of shape [n_samples] (optional)
        Labels constrain the permutation among groups of samples with
        a same label.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.

    verbose : integer, optional
        The verbosity level.

    Returns
    -------
    score : float
        The true score without permuting targets.

    permutation_scores : array, shape = [n_permutations]
        The scores obtained for each permutations.

    pvalue : float
        The returned value equals p-value if `score_func` returns bigger
        numbers for better scores (e.g., accuracy_score). If `score_func` is
        rather a loss function (i.e. when lower is better such as with
        `mean_squared_error`) then this is actually the complement of the
        p-value:  1 - p-value.

    Notes
    -----
    This function implements Test 1 in:

        Ojala and Garriga. Permutation Tests for Studying Classifier
        Performance.  The Journal of Machine Learning Research (2010)
        vol. 11

    """
    X, y = indexable(X, y)
    cv = _check_cv(cv, X, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)
    
    if data_train is None:
        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        # Default behavior of sklearn permutation score
        score = _permutation_test_score(clone(estimator), X, y, cv, scorer)
        permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_permutation_test_score)(
                clone(estimator), X, _shuffle(y, labels, random_state), cv,
                scorer)
            for _ in range(n_permutations))
    else:
        # Modification for 2pn
        # First get the real score, train on nii_optional (actor), test on nii_func (observer)
        score = []
        for train, test in cv:
            estimator.fit(data_train[train], y[train])
            score.append(scorer(estimator, X[test], y[test]))
        score = np.mean(score)
        # Then, get the prmutation scores
        permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_permutation_test_score)(
                clone(estimator), X, _shuffle(y, labels, random_state), cv,
                scorer, data_train)
            for _ in range(n_permutations))
                
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
    return score, permutation_scores, pvalue
    
    
def _permutation_test_score(estimator, X, y, cv, scorer,
                            data_train=None):
    """Auxiliary function for permutation_test_score"""
    avg_score = []
    for train, test in cv:
        if data_train is None:
            estimator.fit(X[train], y[train])
        else:
            estimator.fit(data_train[train], y[train])
        avg_score.append(scorer(estimator, X[test], y[test]))
    return np.mean(avg_score)
    
    
def _shuffle(y, labels, random_state):
    """Return a shuffled copy of y eventually shuffle among same labels."""
    if labels is None:
        ind = random_state.permutation(len(y))
    else:
        ind = np.arange(len(labels))
        for label in np.unique(labels):
            this_mask = (labels == label)
            ind[this_mask] = random_state.permutation(ind[this_mask])
    return y[ind]
