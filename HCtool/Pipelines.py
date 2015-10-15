from __future__ import division

import numpy as np

from nilearn.image import resample_img

from HCtool.utils import load_data
from HCtool import Analysis
from HCtool import Permutations

import HCtool.Searchlight2pn as slight
from HCtool.CCAGFA import CCAGFA


def SingleSubject(cfg):
    '''
    Pipeline to run basic signle-subject classification
    '''

    ## Get crossval scheme, can modify here later if we have variable schemes
    crossval = Analysis.GetCrossval(**cfg)
    cfg.update(crossval=crossval)    
    print('Crossval  acquired')

    ## Load the data
    data_test = load_data(cfg)
    cfg.update(data_test=data_test.get('nii_func'),
               nii_mean=data_test.get('nii_mean'),
               nii_mask=data_test.get('nii_mask'),
               masker=data_test.get('masker'))
    print('Data loaded')
    
    ## CLassify
    cv_scores,y_pred_all = Analysis.Classify(**cfg)
    coef_img,coef_2mm = Analysis.GetImportanceMap(**cfg)    
    print('Classification done')

    ## Searchlight
    sl_4mm = slight.GetSearchLight(**cfg)
    # Upsample
    sl_2mm = resample_img(sl_4mm,target_affine = np.diag((2,2,2)))
    print('Searchlight done')
    
    ##  Permutations
    null_cv_scores = Permutations.GetPermutations(**cfg)
    null_plot = Permutations.plot_permutation(null_cv_scores)       
    print('Permutations done and plotted')

    ## Confmat
    cm = Analysis.GetConfusionMatrix(cfg.get('labels').get('regressor'),y_pred_all)
    cm_plot = Analysis.plot_confusion_matrix(cm)    
    print('Confmat done and plotted')

    ## Cook results
    results = dict(accuracy = np.mean(cv_scores),
                   impmap_4mm = coef_img, impmap_2mm = coef_2mm,
                   searchlight_4mm = sl_4mm, searchlight_2mm = sl_2mm,
                   permutation_score = null_cv_scores[1],
                   permutation_plot = null_plot,
                   confusion_matrix = cm,
                   confusion_matrix_plot = cm_plot)
    print('Returning results')
    return results
    
def Hyperclass(cfg):
    '''
    Pipeline to run 2pn hyperclassification
    '''    
    
    ## Get crossval scheme, can modify here later if we have other schemes
    crossval = Analysis.get_crossval(**cfg)
    cfg.update(crossval=crossval)   
    print('Crossval  acquired')
    
    ## Load the data
    data_train,data_test = load_data(cfg)
    cfg.update(data_train=data_train.get('nii_func'),
               data_test=data_test.get('nii_func'),
               nii_mean=data_test.get('nii_mean'),nii_mask=data_test.get('nii_mask'),
               masker=data_test.get('masker'))
    print('Data loaded')
               
    ## Searchlight before (will be needed to characterize transformation later)
    sl_4mm_before = slight.get_searchlight(**cfg)
    # Upsample
    sl_2mm_before = resample_img(sl_4mm_before,target_affine = np.diag((2,2,2)))
    print('Searchlight before done')
                    
    ## do GFA, if still doesn't work, check dimensions of data in R
    CCApredictor = CCAGFA(userlibpath='/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/R',
                          packages=['bitops','R.methodsS3','R.oo','rmatio','CCAGFA','glmnet','R.matlab','R.methodsS3','R.utils'],
                          repos='http://mirrors.dotsrc.org/cran/')
    # Load all the packages, should always be done first
    CCApredictor.check_r_packages(install=False)
    # Get transformed data, we transform nii_func
    predicted_data = CCApredictor.fit_predict(data_test = cfg.get('data_test'),
                                              data_train = cfg.get('data_train'),
                                              crossval = cfg.get('crossval'),
                                              K=50,save=0,ccafilepath=None)
    # Update the transmitter nii with transformed one
    cfg.update(data_test = predicted_data)
    print('CCA prediction done')
    
    ## Searchlight after, to see the difference after transformation
    sl_4mm_after = slight.get_searchlight(**cfg)
    # Upsample
    sl_2mm_after = resample_img(sl_4mm_after,target_affine = np.diag((2,2,2)))
    print('Searchlight after done')
                
    ## CLassify
    cv_scores,y_pred_all = Analysis.classify(**cfg)
    coef_img,coef_2mm = Analysis.get_impmap(**cfg)    
    print('Classification done')

    ##  Permutations
    null_cv_scores = Permutations.get_permutations(**cfg)
    null_plot = Permutations.plot_permutation(null_cv_scores)         
    print('Permutations done and plotted')
            
    ## Confmat
    cm = Analysis.get_confmat(cfg.get('labels').get('regressor'),y_pred_all)
    cm_plot = Analysis.plot_confusion_matrix(cm)    
    print('Confmat done and plotted')

    ## Cook results
    results = dict(accuracy = np.mean(cv_scores),
                   impmap_4mm = coef_img, impmap_2mm = coef_2mm,
                   searchlight_before_4mm = sl_4mm_before, 
                   searchlight_before_2mm = sl_2mm_before,
                   searchlight_after_4mm = sl_4mm_after, 
                   searchlight_after_2mm = sl_2mm_after,
                   permutation_score = null_cv_scores[1],
                   permutation_plot = null_plot,
                   confusion_matrix = cm,
                   confusion_matrix_plot = cm_plot)
    print('Returning results')
    return results
    
def Betweenclass(cfg):
    '''
    Pipeline to run 2pn between-subject classification    
    '''
    
    ## Get crossval scheme, can modify here later if we have variable schemes
    crossval = Analysis.GetCrossval(**cfg)
    cfg.update(crossval=crossval)  
    print('Crossval  acquired')
    
    ## Load the data
    data_train,data_test = load_data(cfg)
    cfg.update(data_train=data_train.get('nii_func'),
               data_test=data_test.get('nii_func'),
               nii_mean=data_test.get('nii_mean'),nii_mask=data_test.get('nii_mask'),
               masker=data_test.get('masker'))          
    print('Data loaded')
               
    ## CLassify
    cv_scores,y_pred_all = Analysis.Classify(**cfg)
    coef_img,coef_2mm = Analysis.GetImportanceMap(**cfg)   
    print('Classification done')
    
    ##  Permutations
    null_cv_scores = Permutations.GetPermutations(**cfg)
    null_plot = Permutations.plot_permutation(null_cv_scores)         
    print('Permutations done and plotted')
    
    ## Confmat
    cm = Analysis.GetConfusionMatrix(cfg.get('labels').get('regressor'),y_pred_all)
    cm_plot = Analysis.plot_confusion_matrix(cm)
    print('Confmat done and plotted')

    ## Cook results
    results = dict(accuracy = np.mean(cv_scores),
                   impmap_4mm = coef_img, impmap_2mm = coef_2mm,
                   permutation_score = null_cv_scores[1],
                   permutation_plot = null_plot,
                   confusion_matrix = cm,
                   confusion_matrix_plot = cm_plot)
    print('Returning results')
    return results                        
        
                


        



    
    
