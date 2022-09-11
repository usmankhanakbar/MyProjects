# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 22:55:03 2022

@author: dell
"""
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Dense,Conv2DTranspose,Conv2D

class Transferlearning:
    def __init__(self):
        self.shape = Input(shape=(27, 27, 65))  
        self.model = EfficientNetB7(include_top=False,weights = None,input_tensor=self.shape)
  

    def customized(self):
        model = self.model
    
        output = model.layers[-13].output
        output2 = Conv2DTranspose(3,(2,2),strides=(1,1),padding='valid')(output)
        output2 = Conv2DTranspose(9,(4,4),strides=(1,1),padding='valid')(output2)
        output2 = Conv2DTranspose(18,(8,8),strides=(1,1),padding='valid')(output2)
        output2 = Conv2DTranspose(27,(16,16),strides=(1,1),padding = 'valid',activation='tanh')(output2)
        
       
        return model,output2
        

    
    
    
    
    
    