# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 20:53:18 2015

@author: dmitrysmirnov
"""

from __future__ import division
import subprocess

import pickle

from HCtool import Pipelines, Saver
from HCtool.utils import makepath


class ParallelizedAnalysis(object):
    ''' 
    Class to handle interfacing with triton cluster.
    Creates folder for job outputs and keeps parameters for job creation.
    Assumes that analysis is done by executing initfunc on cfg dictionary.
    
    Input
    outdir : path to folder where job and log is saved.
        example : /scratch/braindata/project_folder/specific_analysis
    partition : cluster-specific, short if < 4hr, batch if more.
    t : time estimate, see slurm documentation for format
    mem : memory per cpu
    initfunc : function that is launched on cluster nodes.
    '''
    def __init__(self,outdir=None,
                 partition='short',t='4:0:0',mem='15000',
                 initfunc='initCLSpipeline.py',
                 **kwargs):
        ''' Construct object for interfacing with cluster '''
        makepath(outdir)
        self.outdir = outdir
        self.jobname = '{0}clsjob'.format(outdir)
        self.logfile = '{0}clslog'.format(outdir)
        self.cfgname = '{0}cfg.pickle'.format(outdir)
        self.partition = partition
        self.t = t
        self.mem = mem
        self.initfunc = initfunc
    
    def pickle_cfg(self,cfg):
        '''
        Pickle cfg to use it later with initfunc. Writes pickled file to
        outdir/cfg.pickle
        
        Input
        cfg : dictionary with all obligatory parameters.
        '''        
        with open(self.cfgname, 'wb') as handle:
            pickle.dump(cfg, handle)
            
    def make_job(self,cfg):
        ''' 
        Writes job submission file.
        '''
        with open(self.jobname,'w') as f:
            f.write('#!/bin/sh\n');
            f.write('#SBATCH -p {0}\n'.format(self.partition));
            f.write('#SBATCH -t {0}\n'.format(self.t));
            f.write('#SBATCH -o "{0}"\n'.format(self.logfile));
            f.write('#SBATCH --mem-per-cpu={0}\n'.format(self.mem));
            f.write('module load triton/python/2.7.6\n');
            f.write('cd {0}\n'.format(cfg.get('toolboxpath')));
            f.write('export PATH=/triton/becs/scratch/braindata/shared/GraspHyperScan/tempanaconda/anaconda2_hctool/bin:$PATH\n'); # activate correct anaconda
            f.write('source activate hctool_env\n');
            f.write('python {0} {1}'.format(self.initfunc,self.cfgname));    
    
    def submit_job(self):
        '''
        Interface for submission, executes system call that submits job
        '''
        print('Calling the sbatch {0}'.format(self.jobname))
        subprocess.call('sbatch {0}'.format(self.jobname), shell=True)
        
        
def select_pipeline(cfgname):
    '''
    Function to run from nodes on clusters, loads pickled cfg and runs 
    an analysis based on pathway, and finally saves the results.
    Also can be used to run the analysis on desktop, 
    when no cluster is available, but still requires pickled cfg.
    
    Parameters:
    cfgname - path to config file
    '''
    # Load params
    with open(cfgname, 'rb') as handle:
        cfg = pickle.load(handle)
    #### Pathway specific part starts here
    if cfg['pathway'] == 1:
        results = Pipelines.SingleSubject(cfg)
    elif cfg['pathway'] == 0:
        results = Pipelines.Hyperclass(cfg)
    elif cfg['pathway'] == 2:
        results = Pipelines.Betweenclass(cfg)
    elif cfg['pathway'] == 3:
        results = Pipelines.Sonya_hyperclass(cfg)
    #### Save results in a picke object
    print('Results returned, saving them to {0}'.format(cfg.get('outdir')))
    results.update(filepath=cfg.get('outdir'))
    Saver.save(**results)
    print('Results saved')
    

    
    
