
from __future__ import division

import numpy as np
from nilearn import datasets
from sklearn.linear_model import LogisticRegression

from HCtool.MapReduce import MapReduce

# IMPORTANT: check the load_data function, the path to data is hardcoded there: 
# fname = '{0}/{1}/run{2}/bramila/bramila/epi_STD_mask_detrend_fullreg.nii'.format(dataroot,subject,r+1)
# IMPORTANT2: regressor processing is hardcoded, but can be dropped, 
# if three things are supplied: NTR for each run, regressor for each run after zeros are dropped,
# and to_drop_zeros, which codes TRs when things happen with 1, and rest timepoints with 0

# Initialize params
'''
pathway options,
1 - single-subject analysis
0 - hyperclassification
2 - between-subject analysis

Other params:
TR - samping rate of fMRI
nrun - runs in which data was collected
ncond - number of different labels
convthresh - threshold for dropping time-points after convolving regressor with HRF
downsample - to downsample or not
target_affine - downsampling matrix
anat - where to take affines from
partition - name of slurm partition
t - time for execution of the job
mem - how much memory is expected to be used
toolboxpath - where are the modules
initfunc - how is python called on cluster
dataroot - where all the analyses happens
maskpath - path to mask
maskname - name of the mask
subject_train - optional, if 2pn - compulsory
subject_test - the folder where data is (see comment above about hardcoded filepath)
regressorpath - where is the regressor
n_permutations - how many permutations to test the cls
radius - searchlight
cls - classifier object from sklearn
'''
cfg = dict(TR = 2, nrun = 5, ncond = 4, convthresh = 0.8,
downsample = 1, target_affine = np.diag((4,4,4)),
anat = datasets.load_mni152_template(),
pathway = 0,
partition='short',t='04:00:00',mem='30000',
toolboxpath = '/triton/becs/scratch/braindata/shared/GraspHyperScan/HCtool',
initfunc = 'initCLSpipeline.py',
dataroot = '/triton/becs/scratch/braindata/shared/GraspHyperScan', 
maskpath = '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/ns/grasp.nii',
maskname = 'grasp',
subject_train = 'Sonya_Actor', # optional for HC, used only for training
subject_test = 'Sonya_Observer', # used always in testing, in single subject used for both
regressorpath = '/triton/becs/scratch/braindata/shared/GraspHyperScan/regressorActor.mat', # This needs to be changed, so that we can run thing even when we use some other way of regressor processing
n_permutations = 100,
radius = 6,
cls = LogisticRegression())
##########
OBSERVERS = ('Sonya_Observer',
	         'Sonya_Observer_2',
             'Sonya_Observer_3',
             'Sonya_Observer_4',
             'Sonya_Observer_5',
             'Sonya_Observer_6', 
             'Sonya_Observer_7',
             'Sonya_Observer_8',
             'Sonya_Observer_9',
             'Sonya_Observer_10',
             'Sonya_Observer_11', 
             'Fanny_Observer', 
             'Fanny_Observer_1',
             'Fanny_Observer_2',
             'Fanny_Observer_3')

ACTORS = ('Sonya_Actor', 'Fanny_Actor')

MASKNAMES = ('grasp',
             'LOC',
             'MOTOR',
             'cingularROI',
             'temporalROI',
             'FrontalPole',
             'loc_overlap2_clu')

MASKS = ('/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/ns/grasp.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/LOC.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/MOTOR.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/fsl/ROIvalidation/cingulum/cingularROI.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/fsl/ROIvalidation/temporal/temporalROI.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/fsl/ROIvalidation/FrontalPole.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/localizer/loc_overlap2_clu.nii')
  

for count,elem in enumerate(OBSERVERS):
    cfg.update(subject_test = elem)
    if count < 11:
        cfg.update(subject_train = ACTORS[0])
    else:
        cfg.update(subject_train = ACTORS[1])
    for count_m,elem_m in enumerate(MASKS):
	cfg.update(maskname = MASKNAMES[count_m])
	cfg.update(maskpath = MASKS[count_m])
        # Prepare and submit
        Job = MapReduce(**cfg)
        Job.MakeJob(cfg)
        Job.SubmitJob()















