# HCtool
Hyperclassification tools for two-person neuroscience. This toolbox is used to perform cross-individual classification of fMRI data with additional functional realignment step. For example, we have a person, who was picking up either a ball, or a pen from table, while being scanned in fMRI. We take brain image every two seconds. Another person observed, how the first person was performing actions, and was also scanned in fMRI. The goal for us is to check, whether brain activity of actor (person who picked objects) is similar enough to brain activity of observer, so that machine learning algorithm trained on brain data of actor, can recognize which action was observed from observer's brain data.


I use numpy, nilearn, sklearn and nibabel for data preprocessing, Bayesian CCA from CCAGFA R package, interfaced through rpy2 to perform functional realignment, and logistic regression from sklearn for classification. The whole pipeline is parallelized using Triton cluster (Slurm).


First of all, make sure you have latest version of R installed. Hyperalignment uses some bits of R code. You can get latest version here, depending on the platform: http://cran.r-project.org/

The toolbox can be used in any environment, but this particular example is done on Triton (Science-IT), in Aalto University.

Currently to make it work properly in Triton environment, I use following solution:

1) Go to either frontend or to one of the interactive nodes:
ssh -X username@triton.aalto.fi 


2) Load the settings:
module load triton/python/2.7.6


3) Go to folder, where you want to keep your virtual environment, and install it:

virtualenv venv


4) To begin using it:

source venv/bin/activate


To get the necessary packages:

pip install -r req.txt

# How to use

1) All parameters are defined in nilearnDima.py, to run basic version, just look up what each parameter means and replace with value that fits your case.

2) Pipelines.py contains the example scripts of analysis, there are currently three versions:

-single-subject classification

-between-subject classification

-hyperclassification with BCCA realignment

Pipelines can be changed, or new pipelines added. "pathway" parameter is used to select pipeline.
Pipeline is selected in parallel.select_pipeline function.

3) If running locally, or on different kind of cluster - don't use Job.job_submit method, and instead call the pipeline directly by running parallel.select_pipeline(path_to_pickled_cfg).
