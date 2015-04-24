"""
The searchlight is a widely used approach for the study of the
fine-grained patterns of information in fMRI analysis, in which
multivariate statistical relationships are iteratively tested in the
neighborhood of each location of a domain.
"""
# Authors : Vincent Michel (vm.michel@gmail.com)
#           Alexandre Gramfort (alexandre.gramfort@inria.fr)
#           Philippe Gervais (philippe.gervais@inria.fr)
#
# License: simplified BSD

# Adjustments made by Dmitry Smirnov to allow selection of training and testing tests

import time
import sys
import warnings
from distutils.version import LooseVersion

import numpy as np

import sklearn
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.base import BaseEstimator
from sklearn import neighbors

import nibabel

from nilearn import masking
from nilearn._utils import as_ndarray

ESTIMATOR_CATALOG = dict(svc=svm.LinearSVC, svr=svm.SVR)

def search_light(X, y, estimator, A, scoring=None, cv=None, n_jobs=-1,
                 verbose=0, data_train = None):
    """Function for computing a search_light

    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it taks as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : int, optional
        The verbosity level. Defaut is 0

    Returns
    -------
    scores : array-like of shape (number of rows in A)
        search_light scores
    """
    group_iter = GroupIterator(A.shape[0], n_jobs)
    scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_group_iter_search_light)(
            A.rows[list_i],
            estimator, X, y, scoring, cv,
            thread_id + 1, A.shape[0], verbose,
            data_train)
        for thread_id, list_i in enumerate(group_iter))
    return np.concatenate(scores)

    
class GroupIterator(object):
    """Group iterator

    Provides group of features for search_light loop
    that may be used with Parallel.

    Parameters
    ----------
    n_features : int
        Total number of features

    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'. Defaut is 1
    """
    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        for list_i in split:
            yield list_i


def _group_iter_search_light(list_rows, estimator, X, y,
                             scoring, cv, thread_id, total, verbose=0,
                             data_train=None):
    """Function for grouped iterations of search_light

    Parameters
    -----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    scoring : string or callable, optional
        Scoring strategy to use. See the scikit-learn documentation.
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross validation is
        used or 3-fold stratified cross-validation when y is supplied.

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    verbose : int, optional
        The verbosity level. Defaut is 0

    Returns
    -------
    par_scores : numpy.ndarray
        score for each voxel. dtype: float64.
    """
    par_scores = np.zeros(len(list_rows))
    t0 = time.time()
    for i, row in enumerate(list_rows):
        kwargs = dict()
        if not LooseVersion(sklearn.__version__) < LooseVersion('0.15'):
            kwargs['scoring'] = scoring
        elif scoring is not None:
            warnings.warn('Scikit-learn version is too old. '
                          'scoring argument ignored', stacklevel=2)
        if data_train is None:
            par_scores[i] = np.mean(cross_val_score(estimator, X[:, row],
                                                    y, cv=cv, n_jobs=1,
                                                    **kwargs))
        else:
            # Here goes my modification
            cv_scores = []
            for train, test in cv:
                tempdat_train = data_train[train,:]
                tempdat_train = tempdat_train[:,row]
                # Now the estimator
                estimator.fit(tempdat_train, y[train])
                # Again broadcasting
                tempdat_test = X[test,:]
                tempdat_test = tempdat_test[:,row]
                yhat = estimator.predict(tempdat_test)
                cv_scores.append(np.sum(yhat == y[test]) / float(np.size(y[test])))
            par_scores[i] = np.mean(cv_scores)

        if verbose > 0:
            # One can't print less than each 10 iterations
            step = 11 - min(verbose, 10)
            if (i % step == 0):
                # If there is only one job, progress information is fixed
                if total == len(list_rows):
                    crlf = "\r"
                else:
                    crlf = "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100. - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    "Job #%d, processed %d/%d voxels "
                    "(%0.2f%%, %i seconds remaining)%s"
                    % (thread_id, i, len(list_rows), percent, remaining, crlf))
    return par_scores
                                                
        
##############################################################################
### Class for search_light ###################################################
##############################################################################
class SearchLight(BaseEstimator):
    """Implement search_light analysis using an arbitrary type of classifier.

    Parameters
    -----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False

    Notes
    ------
    The searchlight [Kriegeskorte 06] is a widely used approach for the
    study of the fine-grained patterns of information in fMRI analysis.
    Its principle is relatively simple: a small group of neighboring
    features is extracted from the data, and the prediction function is
    instantiated on these features only. The resulting prediction
    accuracy is thus associated with all the features within the group,
    or only with the feature on the center. This yields a map of local
    fine-grained information, that can be used for assessing hypothesis
    on the local spatial layout of the neural code under investigation.

    Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
    Information-based functional brain mapping.
    Proceedings of the National Academy of Sciences
    of the United States of America,
    vol. 103, no. 10, pages 3863-3868, March 2006
    """

    def __init__(self, mask_img, process_mask_img=None, radius=2.,
                 estimator='svc',
                 n_jobs=1, scoring=None, cv=None,
                 verbose=0):
        self.mask_img = mask_img
        self.process_mask_img = process_mask_img
        self.radius = radius
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose

    def fit(self, imgs, y, data_train=None):
        """Fit the searchlight

        Parameters
        ----------
        img : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            4D image.

        y : 1D array-like
            Target variable to predict. Must have exactly as many elements as
            3D images in img.

        Attributes
        ----------
        `scores_` : numpy.ndarray
            search_light scores. Same shape as input parameter
            process_mask_img.
        """

        # Compute world coordinates of all in-mask voxels.
        mask, mask_affine = masking._load_mask_img(self.mask_img)
        mask_coords = np.where(mask != 0)
        mask_coords = np.asarray(mask_coords + (np.ones(len(mask_coords[0]),
                                                        dtype=np.int),))
        mask_coords = np.dot(mask_affine, mask_coords)[:3].T

        # Compute world coordinates of all in-process mask voxels
        if self.process_mask_img is None:
            process_mask = mask
            process_mask_coords = mask_coords
        else:
            process_mask, process_mask_affine = \
                masking._load_mask_img(self.process_mask_img)
            process_mask_coords = np.where(process_mask != 0)
            process_mask_coords = \
                np.asarray(process_mask_coords
                           + (np.ones(len(process_mask_coords[0]),
                                      dtype=np.int),))
            process_mask_coords = np.dot(process_mask_affine,
                                         process_mask_coords)[:3].T

        clf = neighbors.NearestNeighbors(radius=self.radius)
        A = clf.fit(mask_coords).radius_neighbors_graph(process_mask_coords)
        del process_mask_coords, mask_coords
        A = A.tolil()

        # scores is an 1D array of CV scores with length equals to the number
        # of voxels in processing mask (columns in process_mask)
        X = masking._apply_mask_fmri(imgs,
                nibabel.Nifti1Image(as_ndarray(mask, dtype=np.int8),
                                    mask_affine))

        estimator = self.estimator
        if isinstance(estimator, basestring):
            estimator = ESTIMATOR_CATALOG[estimator]()
        # From here starts Dima's modifications, added nii_optional argumetn to .fit method
        if data_train is None:
            scores = search_light(X, y, estimator, A,
                                  self.scoring, self.cv, self.n_jobs,
                                  self.verbose)
        else:
            scores = search_light(X, y, estimator, A,
                                  self.scoring, self.cv, self.n_jobs,
                                  self.verbose, data_train)
            
        scores_3D = np.zeros(process_mask.shape)
        scores_3D[process_mask] = scores
        self.scores_ = scores_3D
        return self
        
def GetSearchLight(nii_mask,radius,masker,
                   cls,crossval,labels,data_test,data_train=None,
                   **kwargs):
    ''' 
    Construct the searchlight object, run the corresponding analysis.
    Params:
    
    Output:
    '''
    sl_object = SearchLight(mask_img=nii_mask,
                            radius=radius,
                            process_mask_img=nii_mask,
                            estimator=cls,
                            cv=crossval,
                            n_jobs = -1,verbose=1)
    # Searchlight uses 3D images, here we transform the data_test to 3D
    imgs = masker.inverse_transform(data_test)
    y = labels.get('regressor')
    if data_train is None:
        sl_object.fit(imgs = imgs, y = y)
    else:   
        sl_object.fit(imgs = imgs, data_train = data_train, y = y)
    # Extract 3D image and convert to nii
    sl_image = nibabel.Nifti1Image(sl_object.scores_,nii_mask.get_affine())
    return sl_image
    
    
    
    
    
   
   
    