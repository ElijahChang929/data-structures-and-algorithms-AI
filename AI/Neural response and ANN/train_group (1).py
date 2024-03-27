
#%%

import os
import numpy as np
from PIL import Image
import matplotlib as plt
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D,BatchNormalization, MaxPooling2D, DepthwiseConv2D, Activation, GaussianNoise, LocallyConnected1D, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt 
import h5py

#%%
nb_train_samples = 21400
nb_validation_samples = 556



#%%
data = np.load('../Rsp.npy')
test = np.load('../valRsp.npy')

n = nb_train_samples
train_x1 = np.load('../train_r8.npy')

val_x1 = np.load('../val_r8.npy')

#%%
n1 = 9*9*256
STD1 = np.std(train_x1)


train_x = np.reshape(train_x1/STD1,(n,n1))

val_x = np.reshape(val_x1/STD1,(nb_validation_samples,n1))
#%%
fnum = 128*128
ROI = np.mean(test**2,0)>0
roi = np.reshape(ROI,(fnum))


#%%
def mean_squared_error_noise(y_true, y_pred):
    return K.mean(K.square(K.relu(K.abs(y_pred - y_true)-0.0)), axis=-1)

Cnum = 256
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

#alexnet = load_model('alexnet.h5')
model = Sequential()
model.add(Dropout(0.1,input_shape=(n1,)))
model.add(Dense(Cnum, kernel_regularizer=regularizers.l1(1e-5/Cnum), activation='elu'))
model.add(Dense(fnum, use_bias=True))


w1 = model.layers[-1].get_weights()
w2 = np.zeros((Cnum, fnum))
w3 = np.zeros((fnum))
w3[roi] = 0.001
for i in range(Cnum):
    w2[i,:] = w3

w1[0] = w2
model.layers[-1].set_weights(w1)

Adam = optimizers.Adam(lr=2e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss=mean_squared_error_noise, optimizer=Adam, metrics=['mse'])
model.summary()
model.save_weights('initial.hdf5')

#%%

train_y = np.reshape(data,(n,128*128))
val_y   = np.reshape(test,(nb_validation_samples,128*128))

model.load_weights('initial.hdf5')

filepath="Cell.hdf5"

earlyStopping=EarlyStopping(monitor='val_mean_squared_error', patience=50, verbose=1, mode='auto')
saveBestModel = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='auto')


callbacks_list = [earlyStopping,saveBestModel]

model.fit(train_x, train_y, epochs=1600, batch_size=20, validation_data=(val_x, val_y), callbacks=callbacks_list)

#%% test trained model

#model.load_weights(filepath)

pred = model.predict(val_x)

pred1 = np.reshape(pred,(nb_validation_samples,128,128,1))
val_y1 = np.reshape(val_y,(nb_validation_samples,128,128,1))

R = np.zeros((128,128))
VE = np.zeros((128,128))
for i in range(128):
    for j in range(128):
        if np.sum(np.abs(val_y1[:,i,j,0]))>0:
            u2=np.zeros((2,nb_validation_samples))
            u2[0,:]=np.reshape(pred1[:,i,j,0],(nb_validation_samples))
            u2[1,:]=np.reshape(val_y1[:,i,j,0],(nb_validation_samples))
        
        
                
            c2=np.corrcoef(u2)
            R[i,j] = c2[0,1]
            VE[i,j] = 1-np.var(pred1[:,i,j,0]-val_y1[:,i,j,0])/np.var(val_y1[:,i,j,0])

np.save('R.npy',R)
np.save('VE.npy',VE)

