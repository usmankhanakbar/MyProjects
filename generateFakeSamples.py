# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:44:27 2022

@author: dell
"""
import numpy as np
from tensorflow.keras.models import Model
import Generator_Reloaded as Gen_
import glob
from pathlib import Path
from numpy.random import randint
class generateFakeSamples:
    def __init__(self,*args):
        if len(args)>0:
            self.index = args[0]
            self.generator = Gen_.Generator_Reloaded
            self.path =   r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples_2\Combined_Arrays"
            self.label = np.ones((224,224,1)) * -1
        if len(args)>1:
            self.index = args[0]
            self.generator = args[1]
            self.path =   r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples_2\Combined_Arrays"
            self.label = np.ones((224,224,1)) * -1
        if len(args)>2:
            self.index = args[0]
            self.generator = args[1]
            self.path = args[2]
            self.label = np.ones((224,224,1)) * -1
        if len(args)>3:
            self.index = args[0]
            self.generator = args[1]
            self.path = args[2]
            self.label = args[3]
        

        else:
            self.generator = Gen_.Generator_Reloaded
            self.path =   r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples_2\Combined_Arrays"
            self.label = np.ones((27,27,1)) * -1
            self.index = 0
            
    def generate_Fake(self):
        trgPath = Path(self.path).glob('**/*.npy')
        trgpaths = []
        for path in trgPath:
            trgpaths.append(str(path))
        
        t_files = np.load(trgpaths[self.index])
        n_samples = 64
        ix = randint(0,t_files.shape[0],n_samples)
        t_files = t_files[ix,:,:]
        t_files = (t_files)/255.0 * 2.0
        t_files = t_files-1.0
        t_files = t_files.reshape((t_files.shape[1],t_files.shape[2],t_files.shape[0]))
        t_files = np.concatenate((t_files,self.label),axis=2)
       
        t_files = np.expand_dims(t_files,axis=0)
        model = self.generator
        
        model.__init__(model)
        model = model.generator_Fake(model)
        
        
        
        prediction = model.predict(t_files)
        t_labels = np.zeros((1,27,27,1))
        return prediction, t_labels
            