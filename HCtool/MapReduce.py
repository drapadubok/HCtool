# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 20:53:18 2015

@author: dmitrysmirnov
"""

from __future__ import division

import pickle

from HCtool import Pipelines, Saver
from HCtool.utils import makepath


class MapReduce(object):
    ''' Class to handle interfacing with triton cluster , can be replaced with whatever'''
    def __init__(self, dataroot = None,
                 subject_test = None, subject_train = None, maskname = None,
                 pathway = 1, outdir = None,
                 partition='short',t='4:0:0',mem='30000',initfunc='initCLSpipeline.py',     
                 **kwargs):
        ''' Construct object for interfacing with cluster '''
        if outdir is None:
            if pathway == 1:
                outdir = '{0}/single_subject/{1}/{2}/'.format(dataroot,subject_test,maskname)
            elif pathway == 0:
                outdir = '{0}/hyper_subject/{1}/{2}/{3}/'.format(dataroot,subject_train,subject_test,maskname)
	    	elif pathway == 2:
                outdir = '{0}/between_subject/{1}/{2}/{3}/'.format(dataroot,subject_train,subject_test,maskname)
        self.outdir = outdir
        makepath(outdir)
        self.jobname = '{0}clsjob'.format(outdir)
        self.logfile = '{0}clslog'.format(outdir)
        self.cfgname = '{0}cfg.pickle'.format(outdir)
        self.partition = partition
        self.t = t
        self.mem = mem
        self.initfunc = initfunc
        
    def MakeJob(self,cfg):
        ''' Utility to write job file and save config, takes parallel_params as input.
        Should save config file, jobname with all necessary params'''
        with open(self.cfgname, 'wb') as handle:
            pickle.dump(cfg, handle)
        with open(self.jobname,'w') as f:
            f.write('#!/bin/sh\n');
            f.write('#SBATCH -p {0}\n'.format(self.partition));
            f.write('#SBATCH -t {0}\n'.format(self.t));
            f.write('#SBATCH -o "{0}"\n'.format(self.logfile));
            f.write('#SBATCH --mem-per-cpu={0}\n'.format(self.mem));
            f.write('module load triton/python/2.7.6\n');
            f.write('cd {0}\n'.format(cfg.get('toolboxpath')));
            f.write('source venv/bin/activate\n');
            f.write('python {0} {1}'.format(self.initfunc,self.cfgname));
    
    def SubmitJob(self):
        ''' Interface for submission, throws in system call that submits job '''
        import subprocess
	print 'Calling the sbatch {0}'.format(self.jobname)
        subprocess.call('sbatch {0}'.format(self.jobname), shell=True)
  
  
def selectPipeline(cfgname):
    '''
    Parameters:
    cfgname - path to config file, use this function to select analysis pathway
    '''
    # Load params
    with open(cfgname, 'rb') as handle:
        cfg = pickle.load(handle)
    #### Pathway specific part starts here
    if cfg['pathway'] == 1:
        # Single subject
        results = Pipelines.SingleSubject(cfg)
        filepath = '{0}/single_subject/{1}/{2}/'.format(cfg['dataroot'],cfg['subject_test'],cfg['maskname'])
    elif cfg['pathway'] == 0:
        # Hyperclass
        results = Pipelines.Hyperclass(cfg)
        filepath = '{0}/hyper_subject/{1}/{2}/{3}/'.format(cfg['dataroot'],cfg['subject_train'],cfg['subject_test'],cfg['maskname'])
    elif cfg['pathway'] == 2:
        # Between subject
        results = Pipelines.Betweenclass(cfg)
        filepath = '{0}/between_subject/{1}/{2}/{3}/'.format(cfg['dataroot'],cfg['subject_train'],cfg['subject_test'],cfg['maskname'])
    #### Save results in a picke object
    print 'Results returned, saving them to {0}'.format(filepath)
    results.update(filepath=filepath)
    Saver.save(**results)
    print 'Results saved'
    

    
    
