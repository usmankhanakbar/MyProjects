# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:27:28 2022

@author: dell
"""
from pathlib import Path
import numpy as np
import glob2 as glob
import cv2 as cv

class rescaling_images:
    def __init__(self):
        self.path = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples_2\Combined_Arrays";
    def Rescaling_Images(self):
        files = Path(self.path).glob('**/*.npy')
        for file in files:
            two_Ds = np.asarray(np.load(str(file)))
            z,w,h = two_Ds.shape
            image_list = []
            for i in range(z):
                vis0 = two_Ds[i,:,:]
                # cv.imshow('previousImage',vis0)
                vis0 = cv.resize(vis0,(27,27))
                cv.imshow('newImage',vis0)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                image_list.append(vis0)
            two_Ds = np.asarray(image_list)
            np.save(file,two_Ds)
                
            
       


# if __name__ == '__main__':
#     RI = rescaling_images()
#     RI.Rescaling_Images()
    