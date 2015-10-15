
from __future__ import division

import numpy as np
from nilearn import datasets
from sklearn.linear_model import LogisticRegression

from HCtool.parallel import ParallelizedAnalysis
from HCtool.utils import fileparts_name, load_labels


    
    
    
# Initialize params
'''
Initializing script to run the typical analysis within Grasp project.
cfg contains various analysis parameters, here are assigned according to 
requirements of Grasp project.


TR : numeric, samping rate of fMRI in seconds

nrun : integer, number of runs in which data was collected, is used 
in cross-validation

ncond : integer, number of different labels to classify

convthresh : float, threshold for dropping time-points after convolving 
regressor with HRF, 0.6 or 0.8 seems good enough

downsample : 1 or 0, to downsample or not

target_affine : downsampling matrix, if going to 4mm, np.diag((4,4,4))

anat : where to take original affines from, we use MNI template

partition : slurm setting: name of slurm partition to submit to, depends on 
local infrastructure

t : slurm setting, time for execution of the job

mem : slurm setting, how much memory is expected to be used

initfunc : function that takes cfg as argument and runs the complete analysis
    
toolboxpath : path to the folder, where initfunc is located, needed to be 
accessible on cluster

dataroot : path to project folder

maskpath : path to mask .nii

niifilename : name of data file

subject_train - optional argument used only in hyperclassification case: when
training is done on one, and testing on another subject. Assumes that data is
in dataroot/subject_train

subject_test - the folder where data is, assumes dataroot/subject_test.

regressorpath : path to file with labels, right now a hack. Assumes certain
structure of a file, however can be replaced with anything that provides
similar output to load_labels.

n_permutations : int - how many permutations to test the classifier

radius : integer, cm - searchlight ragius.

cls : classifier object from sklearn, might conflict with impmap generation.

pathway options,
1 - single-subject analysis
0 - hyperclassification
2 - between-subject analysis

'''
# Examplary cfg that is used for our analysis
cfg = dict(TR = 2, nrun = 5, ncond = 4, convthresh = 0.8,
downsample = 1, target_affine = np.diag((4,4,4)),
anat = datasets.load_mni152_template(),
pathway = 0,
partition='short',t='04:00:00',mem='15000',
toolboxpath = 
'/triton/becs/scratch/braindata/shared/GraspHyperScan/tempanaconda/hctool',
initfunc = 'initCLSpipeline.py',
dataroot = '/triton/becs/scratch/braindata/shared/GraspHyperScan', 
niifilename = 'bramila/bramila/epi_STD_mask_detrend_fullreg.nii',
maskpath = 
'/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/ns/grasp.nii',
subject_train = 'Sonya_Actor',
subject_test = 'Sonya_Observer',
regressorpath = 
'/triton/becs/scratch/braindata/shared/GraspHyperScan/regressorActor.mat',
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

MASKS = ('/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/ns/grasp.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/LOC.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/MOTOR.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/fsl/ROIvalidation/cingulum/cingularROI.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/fsl/ROIvalidation/temporal/temporalROI.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/fsl/ROIvalidation/FrontalPole.nii',
         '/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/masks/localizer/loc_overlap2_clu.nii')
  
count = 0
elem = OBSERVERS[0]
count_m=0
elem_m=MASKS[0]

for count,elem in enumerate(OBSERVERS):
    cfg.update(subject_test = elem)
    # First 11 observers correspond to first actor
    if count < 11:
        cfg.update(subject_train = ACTORS[0])
    else:
        cfg.update(subject_train = ACTORS[1])
        
    # Decouple regressor processing from pipeline
    labels = load_labels(**cfg)
    cfg.update(labels=labels)
    
    # Create a job for each mask    
    for count_m,elem_m in enumerate(MASKS):
        cfg.update(maskpath = elem_m)
        # Cook path for subject and mask specific job
        pathway = cfg.get('pathway')
        maskname = fileparts_name(cfg.get('maskpath'))
        if pathway == 1:
            cfg.update(outdir = '{dataroot}/single_subject/{subject_test}/{0}/'.format(maskname,**cfg))
        elif pathway == 0:
            cfg.update(outdir = '{dataroot}/hyper_subject/{subject_train}/{subject_test}/{0}/'.format(maskname,**cfg))
        elif pathway == 2:
            cfg.update(outdir = '{dataroot}/between_subject/{subject_train}/{subject_test}/{0}/'.format(maskname,**cfg))
        # Submit job
        Job = ParallelizedAnalysis(**cfg)
        Job.make_job(cfg)
        Job.pickle_cfg(cfg)
        Job.submit_job()






