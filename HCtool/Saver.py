from __future__ import division

import numpy as np
import nibabel

from HCtool.utils import makepath

def save(accuracy = None, permutation_score = None,
         permutation_plot = None, confusion_matrix_plot = None,
         confusion_matrix = None,
         impmap_2mm = None, impmap_4mm = None,
         searchlight_2mm = None, searchlight_4mm = None,
         searchlight_before_2mm = None, searchlight_after_2mm = None,
         searchlight_before_4mm = None, searchlight_after_4mm = None,
         filepath = None,db = None, nii_mask = None,
         **kwargs):
    
    if db is None:# Add database to store numeric data
        # Make filepath folder
        makepath(filepath)
        
        ## Save mean accuracy
	with open('{0}accuracy.csv'.format(filepath),'w') as f:
	    f.write('{0},{1:.2f},{2},{3:.2f}'.format('Accuracy',accuracy,'MaxPermAccuracy',np.max(permutation_score)))

        ## Save permutation values
        np.savetxt('{0}permvals.txt'.format(filepath), permutation_score)
        
        ## Save plots  
        permutation_plot.savefig('{0}permutations.eps'.format(filepath), format='eps')
        confusion_matrix_plot.savefig('{0}confusion_matrix.eps'.format(filepath), format='eps')
        
        ## Save confmat numeric
        np.savetxt('{0}confusion_matrix.txt'.format(filepath), confusion_matrix)
       
        ## Save niis
        # Classifier coefficients
        for i in range(impmap_2mm.get_data().shape[-1]):
            fname = '{0}coef{1}_2mm.nii'.format(filepath,i+1)
            nibabel.save(nibabel.Nifti1Image(impmap_2mm.get_data()[:,:,:,i],impmap_2mm.get_affine()),fname)
	    	fname = '{0}coef{1}_4mm.nii'.format(filepath,i+1)
            nibabel.save(nibabel.Nifti1Image(impmap_4mm.get_data()[:,:,:,i],impmap_4mm.get_affine()),fname)
        
        # Searchlight scores
        if searchlight_4mm is not None:
            nibabel.save(searchlight_4mm,'{0}searchlight_4mm.nii'.format(filepath))
            nibabel.save(searchlight_2mm,'{0}searchlight_2mm.nii'.format(filepath))
        if searchlight_before_4mm is not None:
            nibabel.save(searchlight_before_4mm,'{0}searchlight_before_4mm.nii'.format(filepath))
            nibabel.save(searchlight_before_2mm,'{0}searchlight_before_2mm.nii'.format(filepath))
        if searchlight_after_4mm is not None:
            nibabel.save(searchlight_after_4mm,'{0}searchlight_after_4mm.nii'.format(filepath))
            nibabel.save(searchlight_after_2mm,'{0}searchlight_after_2mm.nii'.format(filepath))
        
