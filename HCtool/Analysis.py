from __future__ import division

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nilearn.image import resample_img

from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.metrics import confusion_matrix

plt.ioff()

def Classify(cls,labels,crossval,data_test,pathway,data_train=None,**kwargs):
        ''' Given cls and crossval scheme, get performance and predicted values.
        
        Params:
        cls - classifier
        regressor - labels for each timepoint
        crossval - crossvalidation scheme
        nii_func - data (if two-subject case - data for testing)
        nii_optional - data for training in two-subject case
        
        Output:
        y_pred_all - predicted categories for each time point
        cv_scores - accuracy for each crossvalidation fold 
        '''
        cv_scores = []
        y_pred_all = []
        tempreg = labels.get('regressor')
        for train, test in crossval:
            if pathway == 1:
                cls.fit(data_test[train,:], tempreg[train])
            else:
                cls.fit(data_train[train,:], tempreg[train])
            y_pred = cls.predict(data_test[test,:]) # Test on other data
            cv_scores.append(np.sum(y_pred == tempreg[test]) / float(np.size(tempreg[test])))
            y_pred_all.append(y_pred)
        y_pred_all = np.concatenate(y_pred_all)
        return cv_scores,y_pred_all
        
        
def GetCrossval(nrun,labels,**kwargs):
    '''Get selector for leave one label (run) out crossvalidation'''
    cvl = np.array([])
    for r in range(nrun):
        idxlen = np.where(labels.get('to_drop_zeros')[r])[0].shape # how many timepoints are left over
        cvl = np.append(cvl,np.ones(idxlen,int)*(r+1))
    crossval = LeaveOneLabelOut(cvl)
    return crossval
    

def GetImportanceMap(cls,labels,data_test,masker,**kwargs):
    ''' Importance map is built from dataset used in training.
    
    Params:
    cls - classifier
    nii_func - data
    regressor - labels for timepoints
    masker - masker that is used for inverse transform (must have similar mask as used in data generation)
    
    Output:
    coef_2mm - upsampled coef_img
    coef_img - original 4D importance map, 4th dimension for label categories
    '''
    # Train with all data
    cls.fit(data_test,labels.get('regressor'))
    coef_ = cls.coef_
    coef_img = masker.inverse_transform(coef_)
    # upsample, save importance maps
    target_affine = np.diag((2,2,2))
    coef_2mm = resample_img(coef_img,target_affine=target_affine)
    return coef_img,coef_2mm


def GetConfusionMatrix(y,yhat):
    ''' Get confusion matrix 
    y - regressor
    yhat - predicted labels'''   
    cm = confusion_matrix(y,yhat)
    # Normalize the confusion matrix by row (i.e by the number of samples
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    return cm_normalized    


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    ''' Parameters:
    cm - confusion matrix,
    title - title
    cmap - colormap, default plt.cm.Blues '''
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar(img,ax=ax)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',  # labels along the bottom edge are off
        labelleft='off',
        right='off')
    return fig
