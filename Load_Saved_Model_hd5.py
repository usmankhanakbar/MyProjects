# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:22:32 2022

@author: Usman Khan
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Nadam,SGD,Adam
from tensorflow.keras.models import load_model
import numpy as np
import open3d as o3d
from numpy.random import randint
import os
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity

class ExtractModel:
    def __init__(self,*argv):
        #self.model = load_model("best_model.h5")   
        #self.model = load_model("best_model_2.h5")
        self.model = load_model("ggg_model.h5")
        self.test_data = argv[0]
        self.target_data = argv[1]
    def load_model(self):
        model = self.model
        return  model
    def evaluate(self,model):
        opt = SGD(lr=0.01,momentum=0.9,nesterov=True)
        model.compile(loss = 'mae',optimizer = opt, metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
        test_data = np.load(self.test_data)
        #target_data = np.load(self.target_data)
        n_samples = 64
        ix = randint(0,test_data.shape[0],n_samples)
        t_data = test_data[ix,:,:]
        t_data = np.reshape(t_data,(224,224,64))
        t_data = np.expand_dims(t_data,axis = 0)
        # target_data = np.expand_dims(target_data,axis=0)
        # target_data = np.expand_dims(target_data,axis=3)
        results = model.evaluate(t_data)
        return results
    def predict(self,model):
        #opt = SGD(lr=0.01,momentum=0.9,nesterov=True)
        opt = Adam(learning_rate = 0.0002,beta_1=0.5)
        #model.compile(loss = 'mae',optimizer = opt, metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
        model.compile(loss='binary_crossentropy',optimizer=opt)
        test_data = np.load(self.test_data)
        target_data = np.load(self.target_data)
        n_samples = 64
        ix = randint(0,test_data.shape[0],n_samples)
        t_data = test_data[ix,:,:]
        t_data = np.reshape(t_data,(27,27,64))
        _Ones = np.ones((27,27,1)) * -1
        t_data = np.concatenate((t_data,_Ones),axis=2)
        t_data = np.expand_dims(t_data,axis = 0)
        
        prediction = model.predict(t_data)
       
        return prediction, target_data
    def scale_up(pred):
        inImage = ((pred * 10**9)+ 1)/(2) * (31)
        return inImage
    def writebackpcd(array_p):
       pcd = o3d.geometry.PointCloud()
       pcd.points = o3d.utility.Vector3dVector(array_p)
        
       return pcd
    def displaypcd(pcd):
       o3d.visualization.draw_geometries([pcd])
       
    def calcScores(a_sparse,b_sparse):
        if a_sparse.shape[1]== 3:
            a_sparse = np.transpose(a_sparse)
        if b_sparse.shape[1] == 3:
            b_sparse = np.transpose(b_sparse)
        sim_sparse = cosine_similarity(a_sparse, b_sparse, dense_output=False)
        print(sim_sparse)
        print(sim_sparse.shape)
        
        
        
if __name__ == "__main__":
    # test_data = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples\combined_Target_2.npy"
    # target_data = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TargetSamples\3D_c2.npy"
   # filename = r"E:\Thesis\Databases\Training_Testing_Dataset\combined_trg_1\trained_model_on_c1"
    test_data =  r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples_2\Combined_Arrays\c02.npy"
    target_data =  r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TargetSamples\3D_c2.npy"
    
   
    filename = r"F:\Thesis\Databases\Training_Testing_Dataset\combined_trg_1"
    ExtractModel.__init__(ExtractModel,test_data,target_data)
    
    model = ExtractModel.load_model(ExtractModel)
    #results = ExtractModel.evaluate(ExtractModel,model)
    prediction,target_data = ExtractModel.predict(ExtractModel,model)
    pred = prediction[0,:,:,:]
    pred = ExtractModel.scale_up(pred)
    pred = np.reshape(pred,(6561,3))
    
    pcd = ExtractModel.writebackpcd(pred)
    ExtractModel.displaypcd(pcd)
    # evalResults = ExtractModel.evaluate(ExtractModel,model)
    # print(evalResults)
    
    ExtractModel.calcScores(target_data,pred)
    # radii = [0.005, 0.01]
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([rec_mesh])
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    # o3d.visualization.draw_geometries([mesh])