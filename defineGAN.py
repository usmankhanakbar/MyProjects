# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:13:57 2022

@author: dell
"""
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding,Dense,Concatenate,Conv1D,LeakyReLU,Reshape,concatenate,Reshape,Conv2D,Flatten
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow import expand_dims
from tensorflow.keras.models import Model
from tensorflow.compat.v1 import squeeze

from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import Reshape,Concatenate,Dense
from tensorflow.keras.layers import RepeatVector
import numpy as np

class defineGAN:
    def __init__(self,*args):
        if len(args)==0:
            self.gmodel = None
        elif len(args)>0:
            self.gmodel = args[0]
            self.dmodel = args[1]
            self.truesamples = args[2]
            self.dshape = [6561*3,]
           
        elif len(args)>1:
            self.gmodel = args[0]
            self.dmodel = args[1]
            self.truesamples = args[2]
            self.dshape = [6561*3,]
    def getEmbedding(self,mInput):
         # return Embedding(32,3,input_length = 3)(mInput)
        return Embedding(32,27,input_length=27*27*27)(mInput)
    def defGAN(self):
        self.gmodel.Generator_Reloaded.__init__(self.gmodel)
        im_label_input = self.gmodel.Generator_Reloaded.generator_Fake(self.gmodel).input
        imageInput = im_label_input[:,:,:,0:63]
        gen_label = im_label_input[:,:,:,64]
        fakeOutput = self.gmodel.Generator_Reloaded.generator_Fake(self.gmodel).output
        
        self.dmodel.Discriminator.__init__(self.dmodel,self.dshape)
        f_label = np.zeros((1,27,27,27,1),dtype='float32')
        truesamples = self.defineGAN.getEmbedding(self,self.truesamples)
        f_Output = self.defineGAN.getEmbedding(self,fakeOutput)
        merge = Concatenate(axis=-1)([truesamples,f_Output,f_label])
        gen_label = expand_dims(gen_label,axis=3)
        gen_label = Dense(27)(gen_label)
        ganOutput = self.dmodel.Discriminator.discriminatorFn(self.dmodel)(merge,gen_label)
        #ganOutput = self.dmodel.Discriminator.discriminatorFn(self.dmodel)(merge,gen_label).output
       
        
        model = Model([merge,gen_label],ganOutput)
        opt = Adam(learning_rate = 0.0002,beta_1=0.5)
        model.compile(loss='binary_crossentropy',optimizer=opt)
        return model