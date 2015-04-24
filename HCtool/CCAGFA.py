import numpy as np

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr, quiet_require

from HCtool.utils import makepath

rpy2.robjects.numpy2ri.activate()


class CCAGFA(object):
    ''' Class to store methods and attributes that are used in conjuction with rpy '''
    def __init__(self,
                 userlibpath='/triton/becs/scratch/braindata/shared/GraspHyperScan/Bayes/R/',
                 packages=['bitops','R.methodsS3','R.oo','rmatio','CCAGFA','glmnet','R.matlab','R.methodsS3','R.utils'],
                 repos='http://mirrors.dotsrc.org/cran/'):
        self.userlibpath = userlibpath
        self.packages = packages
        self.repos = repos
        makepath(self.userlibpath)
        self.r = ro.r
        
    def check_r_packages(self,install=False):
        ''' Check installation of packages
        Parameters:
        install - do these packages need to be installed first?'''
        utils = importr('utils')
        for item in self.packages:
            if install:
                utils.install_packages(item,self.userlibpath,self.repos)
            else:
                quiet_require(item,lib_loc = self.userlibpath)
    
    def fit_predict(self,data_test,data_train,crossval,K=50,save=1,ccafilepath=None):
        ''' 
        K - number of components
        save - should we save model, should we save the transformed data?
        Can be improved so that it does fitting useing separate fit method, and predicting with separate predict, but not now
        '''
        self.r('opts <- getDefaultOpts()')
        self.r('opts$verbose=0')
        self.r('K = {0}'.format(K))
        
        predicted_data = np.zeros(data_test.shape, dtype=float)
        
        i = 0
        for train, test in crossval:            
            # Throw the data into R
            ro.globalenv['dobs_train'] = data_test[train,:]
            ro.globalenv['dobs_test'] = data_test[test,:]
            ro.globalenv['dcls_train'] = data_train[train,:]
            ro.globalenv['dcls_test'] = data_train[test,:]
            
            # fit the CCA model and get data prediction
            self.r('ccafit = GFAexperiment(list(dcls_train,dobs_train),K,opts)')
            self.r('ccapred = GFApred(c(0,1),list(0*dobs_test,dobs_test),ccafit,sample=F)$Y[[1]]')
            #predicted_data.append(np.array(self.r.ccapred.rx()))

            ## Modification for testing
            predicted_data[test,:] = np.array(self.r.ccapred.rx())
          
            # Save the model
            if save:
                i = i+1
                self.r.writeMat('{0}_K{1}_cvi{2}.mat'.format(ccafilepath,K,i),ccafit=self.r.ccafit)

        # Combine the data
        #predicted_data = np.concatenate(predicted_data) 
        return predicted_data


#### UNUSED       
#for i in range(crossval.n_unique_labels):
#            data_generator = generate_r_data(data_test,data_train,crossval)
#            dobs_train,dobs_test,dcls_train,dcls_test = data_generator.next()
def generate_r_data(data_test,data_train,crossval):
    ''' Make a generator of data for CCA'''
    for train, test in crossval:
        dobs_train = data_test[train,:]
        dobs_test = data_test[test,:]
        dcls_train = data_train[train,:]
        dcls_test = data_train[test,:]
        yield dobs_train,dobs_test,dcls_train,dcls_test            
            
def generate_cca_model(data_test,data_train):
    ''' Make a generator of CCA models '''
    raise NotImplemented