"""
Helper functions
"""
from os import makedirs
from os.path import normpath,dirname,exists,abspath

import numpy as np
from scipy import stats

from nilearn.image import resample_img, mean_img
from nilearn.input_data import NiftiMasker
from nibabel import load, Nifti1Image

from HCtool.fixdict import loadmat

def LoadData(cfg):
    if cfg.get('pathway') == 1:
        # If single subject case, assume that it is receiver
        cfg.update(subject=cfg.get('subject_test'))
        return load_data(**cfg)
    else:
        # if two subjects, first assume receiver, then transmitter
        # Receiver is always used in testing, transmitter only for training
        cfg.update(subject=cfg.get('subject_test'))
        data_test = load_data(**cfg)
        cfg.update(subject=cfg.get('subject_train'))
        data_train = load_data(**cfg)
        return data_train,data_test
        
        
def LoadLabels(regressorpath,nrun,ncond,TR,convthresh,**kwargs):
        """
        Given parameters in cfg, return labels, selector to drop zeros and NTR
        """
        labels = loadmat(regressorpath)
        NTR = list()
        #separate_column_regressor = list()
        regressor = list()
        to_drop_zeros = list()
        for i in range(nrun):
            tempreg = labels['regressor']['run{0}'.format(i+1)]
            NTR.append(tempreg.shape[0])
            conds_conv = np.empty([NTR[i],ncond],dtype=float)
            for c in range(ncond):
                convreg = np.convolve(spm_hrf(TR),tempreg[:,c])[:NTR[i]]
                convreg = convreg/max(convreg)
                convreg[convreg < convthresh] = 0
                conds_conv[:,c] = np.sign(convreg)
            #separate_column_regressor.append(conds_conv)
            regressor.append(conds_conv.nonzero()[1])#+1
            to_drop_zeros.append(np.sum(conds_conv,axis=1))
        regressor = np.concatenate(regressor)
        return dict(regressor=regressor,to_drop_zeros=to_drop_zeros,NTR=NTR)
        
        
def spm_hrf(TR,p=[6,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution
    Arguments:
    Required:
    TR: repetition time at which to generate the HRF (in seconds)
    Optional:
    p: list with parameters of the two gamma functions:
                                                         defaults
                                                        (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32
       Borrowed implementation from Poldrack's git
    """
    p=[float(x) for x in p]
    fMRI_T = 16.0
    TR=float(TR)
    dt  = TR/fMRI_T
    u   = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf=stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=np.array(range(np.int(p[6]/TR)))*fMRI_T
    hrf=hrf[list(good_pts)]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf = hrf/np.sum(hrf);
    return hrf
    
    
def wrap_mask(nii_img, nii_mask, downsample, target_affine):
    ''' If downsample, downsample, mask and return, otherwise - just mask and return'''
    nii_mask = _check_binary(nii_mask)
    if downsample:
        nii_img = resample_img(nii_img,target_affine=target_affine)
        nii_mask = resample_img(nii_mask,target_affine=target_affine,interpolation='nearest')
    masker = NiftiMasker(nii_mask,standardize=True)
    nii_masked = masker.fit_transform(nii_img)
    return nii_masked, nii_mask, masker
    
    
def _check_binary(img):
    ''' Utility to binarize the mask '''
    timg = img.get_data()
    timg = timg != 0 # binarize
    img = Nifti1Image(timg.astype(np.int),img.get_affine())
    return img    
    
    
def drop_labels(nii,to_drop_zeros):
    """
    Drop the labelless timepoints
    """
    nii = nii[:to_drop_zeros.shape[0],:] # trim to the length of label vector (assuming that file is longer than labels)
    nii = nii[to_drop_zeros==1] # keep only the labeled points
    return nii


def load_data(anat,downsample,target_affine,
             dataroot,subject,maskpath,nrun,
             labels,**kwargs):
    ''' Given parameters in cfg, return masked and concatenated over runs data '''
    nii_func = list()
    for r in range(nrun):
        fname = '{0}/{1}/run{2}/bramila/bramila/epi_STD_mask_detrend_fullreg.nii'.format(dataroot,subject,r+1) # Should remove hardcoding of the path
        # Load all the data
        nii_img = load(fname, mmap=False)
        nii_img.set_sform(anat.get_sform())
        # Get mean
        nii_mean = mean_img(nii_img)
        # Mask
        nii_mask = load(maskpath)
        nii_mask.set_sform(anat.get_sform())
        # Mask the data
        nii_img, nii_mask, masker = wrap_mask(nii_img,nii_mask,downsample,target_affine)
        # Drop zero timepoints, zscore
        nii_img = drop_labels(nii_img,labels.get('to_drop_zeros')[r])
        nii_func.append(stats.zscore(nii_img,axis=0)) # TEST, I want to zscore over time, which is first dimention here
    # throw data together
    nii_func = np.concatenate(nii_func)
    return dict(nii_func=nii_func,nii_mean=nii_mean,masker=masker,nii_mask=nii_mask)


def makepath(path):
    """ creates missing directories for the given path and
        returns a normalized absolute version of the path.
    - if the given path already exists in the filesystem
      the filesystem is not modified.
    - otherwise makepath creates directories along the given path
      using the dirname() of the path. You may append
      a '/' to the path if you want it to be a directory path.
    from holger@trillke.net 2002/03/18
    """
    dpath = normpath(dirname(path))
    if not exists(dpath): makedirs(dpath)
    return normpath(abspath(path))