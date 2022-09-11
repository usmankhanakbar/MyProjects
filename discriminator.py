from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding,Dense,Concatenate,Conv3D,LeakyReLU,Reshape,concatenate,Reshape,Conv2D,Flatten
from tensorflow.keras.activations import sigmoid
from tensorflow import expand_dims
from tensorflow.keras import Model 
from tensorflow.keras.layers import ZeroPadding1D
import numpy as np
from tensorflow.keras.layers import RepeatVector
from tensorflow.compat.v1 import squeeze
class Discriminator:
    
    def __init__(self,*args):
        self.myshape = args[0]
        
    def getInput(self):
        return Input(shape=self.myshape)
    def getLabel(self):
        return Input(shape=[27,27,1])
    def getEmbedding(self,mInput):
       # return Embedding(32,3,input_length = 3)(mInput)
        return Embedding(32,27,input_length=27*27*27)(mInput)
    
    def discriminatorFn(self):
        input_shape = [6561,3,]
        # self.__init__(input_shape)
        intput = self.Discriminator.getInput(self)
        #intput = self.getInput()
        intput = Reshape((27,27,27))(intput)
       
        label = self.Discriminator.getLabel(self)
        label = Dense(27)(label)
        label = expand_dims(label,axis=4)
        # label = self.getLabel()
        #Emdedding = self.Discriminator.getEmbedding(self,intput)
        Emdedding = self.Discriminator.getEmbedding(self,intput)
        inImage = Input(shape = input_shape)
        inImage = Reshape((27,27,27))(inImage)
        
        EmbeddingIn = self.Discriminator.getEmbedding(self,inImage)
        # EmbeddingIn = self.getEmbedding(inImage)
        #inImage = EmbeddingIn
        
        # merge = Concatenate(axis=-1)([intput,inImage,label])
        merge = Concatenate(axis=-1)([Emdedding,EmbeddingIn,label])
       # The input shape will be 81,81,55
        fe = Conv3D(128,1,strides=1,padding='valid')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv3D(64,1,strides=1,padding='valid')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv3D(32,1,strides=1,padding='valid')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv3D(16,1,strides=1,padding='valid')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv3D(1,1,strides=1,padding='valid')(merge)
        
        
        output = sigmoid(fe)
       
        
        model = Model(inputs=merge,outputs=output)
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model
    
    
    
    
    
    
  



# if __name__ == "__main__":
    
#     inputshape = [6561*3,]
#     dI = Discriminator(inputshape)
    
#     # intput = dI.getInput()
#     # Emdedding = dI.getEmbedding(intput)
#     model = dI.discriminatorFn()
#     model.summary()
    #inImage = Input(shape=input_shape)
    #inImage = expand_dims(inImage,axis=2)
#     EmbeddingIn = dI.getEmbedding(dI,inImage)
#     merge = Concatenate(axis=-1)([EmbeddingIn,Emdedding])
#     fe = Conv1D(30,1,strides=1,padding='valid')(merge)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Reshape((600,600))(fe)
#     fe = expand_dims(fe,axis=3)
#     fe = Conv2D(64,(3,3),strides=(2,2),padding='valid')(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Conv2D(32,(3,3),strides=2,padding='valid')(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Conv2D(16,(3,3),strides=2,padding='valid')(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Conv2D(8,(3,3),strides=2,padding='valid')(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Conv2D(4,(3,3),strides=2,padding='valid')(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Conv2D(2,(3,3),strides=2,padding='valid')(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Conv2D(1,(3,3),strides=2,padding='valid')(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Conv2D(1,(3,3),strides=2,padding='valid')(fe)
#     fe = Flatten()(fe)
#     fe = sigmoid(fe)
#     fe