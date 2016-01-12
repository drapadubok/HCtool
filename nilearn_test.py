# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:50:35 2015

@author: smirnod1
"""
from __future__ import division

import os.path

import logging 

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
cfg = dict(TR = 1.7, nrun = 4, ncond = 3, convthresh = 0.6,
downsample = 1, target_affine = np.diag((4,4,4)),
anat = datasets.load_mni152_template(),
pathway = 3,
#partition='short',t='04:00:00',mem='8000',
partition='batch',t='23:00:00',mem='15000',
toolboxpath = 
'/triton/becs/scratch/braindata/shared/GraspHyperScan/tempanaconda/HCtool',
initfunc = 'initCLSpipeline.py',
dataroot = '/triton/becs/scratch/braindata/legarm1/FacialExpressions', 
niifilename = 'bramila/epi_STD_mask_detrend_fullreg.nii',
maskpath = 
'/triton/becs/scratch/braindata/anikins1/FacialExpressionPilot/masks/whole_GM.nii',
subject_train = 's4/Making',
subject_test = 's4/Observing',
regressorpath = 
'/triton/becs/scratch/braindata/anikins1/all the scripts/New_Hyperalignment/regressor_by_dima_make.mat',
n_permutations = 100,
radius = 6,
cls = LogisticRegression())

###
OBSERVERS = [
's4/Observing',
's5/Observing',
's6/Observing',
's7/Observing',
's8/Observing',
's9/Observing',
's10/Observing',
's11/Observing',
's12/Observing',
's13/Observing',
's14/Observing',
's15/Observing']

ACTORS = [
's4/Making',
's5/Making',
's6/Making',
's7/Making',
's8/Making', 
's9/Making',
's10/Making',
's11/Making',
's12/Making',
's13/Making',
's14/Making',
's15/Making']
 
MASKS = [
'/triton/becs/scratch/braindata/anikins1/FacialExpressionPilot/masks/whole_GM.nii',
'/triton/becs/scratch/braindata/anikins1/FacialExpressionPilot/masks/emotions_binmask.nii',
'/triton/becs/scratch/braindata/anikins1/FacialExpressionPilot/masks/faces_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/insula/Insula_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/Amygdala/amygdala_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/anterior_cingulate_gyrus/Anterior_Cingulate_Gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/frontal_medial_cortex/Frontal_Medial_Cortex_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/frontal_orbital_cortex/Frontal_Orbital_Cortex_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/frontal_pole/frontal_pole_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/inferior_frontal_gyrus/Inferior_Frontal_Gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/paracingulate_gyrus/Paracingulate_Gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/postcentral_gyrus/Postcentral_Gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/precentral_gyrus/precentral_gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/posterior_cingulate_gyrus/Posterior_Cingulate_Gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/precuneous_cortex/Precuneous_Cortex_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/Supplementary_Motor_Cortex/Supplementary_Motor_Cortex_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/thalamus/Thalamus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/Hippocampus/Hippocampus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/Accumbens/Accumbens_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/fusiform_gyrus/fusiform_gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/inf_occ_gyrus/Lateral_Occipital_Cortex_inferior_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/STS/Superior_Temporal_Gyrus_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/making_all.nii',
'/triton/becs/scratch/braindata/legarm1/masks/observing_all.nii',
'/triton/becs/scratch/braindata/legarm1/masks/sum_all_binmask.nii',
'/triton/becs/scratch/braindata/legarm1/masks/intersection_all_binmask.nii']

#logging.basicConfig(filename='/triton/becs/scratch/braindata/anikins1/FacialExpressionPilot/hyper_job_submission.log')

for count,elem in enumerate(OBSERVERS):
    cfg.update(subject_test = elem)
    cfg.update(subject_train = ACTORS[count])
    
    # Create a job for each mask    
    for count_m,elem_m in enumerate(MASKS):
        cfg.update(maskpath = elem_m)
        # Cook path for subject and mask specific job
        pathway = cfg.get('pathway')
        maskname = fileparts_name(cfg.get('maskpath'))
        if pathway == 1:
            cfg.update(outdir = '{dataroot}/single_subject/{subject_test}/{0}/'.format(maskname,**cfg))
        elif pathway == 0 or pathway == 3:
            cfg.update(outdir = '{dataroot}/hyper_subject/{subject_train}/{subject_test}/{0}/'.format(maskname,**cfg))
        elif pathway == 2:
            cfg.update(outdir = '{dataroot}/between_subject/{subject_train}/{subject_test}/{0}/'.format(maskname,**cfg))
        # Submit job
        #try:
        #    os.path.isfile(os.path.join(cfg.get('outdir'),'accuracy.csv'))            
        #except Exception, e:
        #    logging.exception(e, exc_info=1)
        if not os.path.isfile(os.path.join(cfg.get('outdir'),'accuracy.csv')):
            Job = ParallelizedAnalysis(**cfg)
            Job.make_job(cfg)
            Job.pickle_cfg(cfg)
            Job.submit_job()

