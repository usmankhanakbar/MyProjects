# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 00:35:38 2022

@author: dell
"""

class Scaling:
    def __init__(self,in_Image):
        self.in_Image = in_Image
       
    def scale_down(self):
        inImage = (self.in_Image - 0)/(31-0)*(2)
        
        inImage = inImage - 1
        return inImage
    
    def scale_up(self):
        inImage = (self.in_Image + 1)/(2) * (31)
        return inImage
    
    def scale_dwn(self):
        inImage = (self.in_Image-0)/(255-0)*(2)
        inImage = inImage-1
        return inImage
    