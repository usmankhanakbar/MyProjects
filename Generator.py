import Decoder
import Encoder
import Clasic_CNN
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam,SGD,Adadelta
import os
from matplotlib.animation import FuncAnimation
from numpy.random import randint
import glob
from time import sleep

import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import Context_Aware_Fusion_Module as CAFM
import refiner as Refiner
from TransferLearning import Transferlearning as TFL
from tensorflow.keras.models import Model


class Generator:
    def __init__(self,*args):
        if len(args)> 0:
            self.npyfilespath = args[0]
        elif len(args)>1:
            self.fpath = args[1]
        elif len(args)>2:
            self.model = args[2]
        else:
            self.filename_input3D = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples_2\Combined_Arrays"
            self.filename_output3D = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TargetSamples"
            self.TFL = TFL
            
    def load_(npyfilespath,fpath):
        os.chdir(npyfilespath)
        dataArray = []
        file_size = os.path.getsize(fpath)
        if file_size == 0:
            with open(fpath, 'wb') as f_handle:
                for npfile in glob.glob("*.npy"):
                    filepath = os.path.join(npyfilespath, npfile)
                    dataArray.append(np.load(filepath))
                dataArray = np.asarray(dataArray)
                np.save(f_handle,dataArray)
                dataArray = np.load(fpath)
        else:
            dataArray = np.load(fpath)
        if len(dataArray)<64:
            dataArray = np.concatenate((dataArray,np.load(fpath)),axis=0)
        
        return dataArray

 
            
            # plt.cla()
          
            # plt.plot(d_loss1)
           
            # plt.tight_layout()

    def generator(self):
        ############################################################
        #Trying Resnet
        # model = Sequential()
        # model = Clasic_CNN.Clasic_CNN.__init__(Clasic_CNN,Sequential())
        # model = Clasic_CNN.Clasic_CNN.VGG16(Clasic_CNN)
        ###########################################################
        tfl = self.TFL
        tfl.__init__(tfl)
        model,output = tfl.customized(tfl)
        model = Model(inputs=model.inputs,outputs=output)
        print(model.summary())
        # model1 = Encoder.Pix2VoxF(model)
        filename_output3D = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TargetSamples"
        filename_input3D = r"F:\Thesis\Databases\Training_Testing_Dataset\TrainingFolder\TrainingSamples_2\Combined_Arrays"
        f_model_json = r"F:\Thesis\Databases\Training_Testing_Dataset\combined_trg_1\model.json"
        f_model_h5 = r"F:\Thesis\Databases\Training_Testing_Dataset\combined_trg_1\model.h5"
        
        history = []
        history_mse = []
       
        output = Encoder.Encoder.Pix2VoxM(output)
        output = Decoder.Decoder.dcdr__(output)
        model2 = Model(inputs= model.inputs,outputs=output)
        model2.summary()
       
        opt = Nadam(lr = 0.001,beta_1=0.9, beta_2 = 0.999,epsilon=1e-07)
        #opt = SGD(lr=0.01,momentum=0.9,nesterov=True)
        #opt = Adadelta(learning_rate=0.01, rho=0.95, epsilon=1e-07)
        model2.compile(loss = 'mae',optimizer = opt, metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
        trgPath = Path(self.filename_input3D).glob('**/*.npy')
        tgtPath = Path(self.filename_output3D).glob('**/*.npy')
        trgpaths = []
        tgtpaths = []
        for path in trgPath:
            trgpaths.append(str(path))
        
        for path in tgtPath:
            tgtpaths.append(str(path))
            
        print(trgpaths)
        print(tgtpaths)
        
        es = EarlyStopping(monitor='mse',mode ='min',verbose=1,patience = 300)
        mc = ModelCheckpoint('best_model_2.h5',monitor='mae',mode='min',verbose=1,save_best_only = True)
       #mc = ModelCheckpoint('best_model.h5',monitor='cosine_proximity',mode='max',verbose=1,save_best_only = True)
        
        for i in range(len(trgpaths)-1):
           trgfile = np.load(trgpaths[i])
           tgtfile = np.load(tgtpaths[i])
           n_samples = 64
           ix = randint(0,trgfile.shape[0],n_samples)
           t_files = trgfile[ix,:,:]
           t_files = t_files.reshape((t_files.shape[1],t_files.shape[2],t_files.shape[0]))
           t_files = np.expand_dims(t_files,axis=0)
           tgtfile = np.expand_dims(tgtfile,axis = 2)
           tgtfile = np.expand_dims(tgtfile,axis = 0)
           H = model2.fit(t_files,tgtfile,batch_size=10,epochs=3000,verbose=1,callbacks=[es,mc])
           #H = model2.fit(t_files,tgtfile,batch_size=1,epochs=1000,verbose=1)
           history.append(H.history["loss"])
           history_mse.append(H.history["mse"])
          
        return model2
  