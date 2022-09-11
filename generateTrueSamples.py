# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:26:43 2022

@author: dell
"""
from pathlib import Path
import numpy as np
from numpy.random import randint
import os
import tensorflow as tf
class generateTrueSamples:
    def __init__(self,*args):
        if len(args)>0:
            self.index = args[0]
            self.path  = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TargetSamples"
        elif len(args)>1:            
            self.index = args[0]
            self.path  = args[1]
      
        
    def generateSamples(self):
        trgpaths = []
        for root,dirs,files in os.walk(self.path):
            for file in files:
                trgpaths.append(root+'/'+file)
                
       
        trgfile = np.load(trgpaths[self.index])
        trgfile = np.asarray(trgfile)
        trgfile = trgfile.reshape((1,27,27,27))
        trgfile = tf.convert_to_tensor(trgfile)
       
        t_labels = np.ones((1,27,27,1))
       
        return trgfile,t_labels