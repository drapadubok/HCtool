#!/usr/bin/python
'''
Utility to run analysis pipeline from command line.
We submit it to slurm.
'''
import sys
from HCtool.MapReduce import selectPipeline
print 'Path to cfg:', str(sys.argv[1])
selectPipeline(sys.argv[1])
