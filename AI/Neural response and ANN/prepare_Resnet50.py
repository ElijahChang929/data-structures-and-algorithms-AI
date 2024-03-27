

import os
import numpy as np
from PIL import Image
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt 
import h5py

#%%

train_x1 = np.load('train_x.npy')
val_x1 = np.load('val_x.npy')

train_x = preprocess_input(train_x1)
val_x = preprocess_input(val_x1)

#%%
from tensorflow.keras.layers import Conv2D,BatchNormalization, MaxPooling2D, DepthwiseConv2D, Activation, GaussianNoise, LocallyConnected2D, Input, Cropping2D
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras import applications
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

#alexnet = load_model('alexnet.h5')
input_tensor = Input(shape=(100, 100, 3))
base = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
#base.summary()

#%%
x = base.layers[1].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_1.npy',train_res)
np.save('val_1.npy',val_res)
#%%
x = base.layers[2].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_2.npy',train_res)
np.save('val_2.npy',val_res)
#%%
x = base.layers[3].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_3.npy',train_res)
np.save('val_3.npy',val_res)
#%%
x = base.layers[4].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_4.npy',train_res)
np.save('val_4.npy',val_res)
#%%
x = base.layers[5].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_6.npy',train_res)
np.save('val_6.npy',val_res)
#%%
x = base.layers[9].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_9.npy',train_res)
np.save('val_9.npy',val_res)
#%%
x = base.layers[12].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_12.npy',train_res)
np.save('val_12.npy',val_res)
#%%
x = base.layers[17].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_17.npy',train_res)
np.save('val_17.npy',val_res)

#%%
x = base.layers[18].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_18.npy',train_res)
np.save('val_18.npy',val_res)
#%%
x = base.layers[21].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_21.npy',train_res)
np.save('val_21.npy',val_res)
#%%
x = base.layers[22].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_22.npy',train_res)
np.save('val_22.npy',val_res)
#%%
x = base.layers[24].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_24.npy',train_res)
np.save('val_24.npy',val_res)
#%%
x = base.layers[27].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_27.npy',train_res)
np.save('val_27.npy',val_res)
#%%
x = base.layers[28].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_28.npy',train_res)
np.save('val_28.npy',val_res)
#%%
x = base.layers[31].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_31.npy',train_res)
np.save('val_31.npy',val_res)
#%%
x = base.layers[34].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_34.npy',train_res)
np.save('val_34.npy',val_res)

x = base.layers[37].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_37.npy',train_res)
np.save('val_37.npy',val_res)
#%%
x = base.layers[38].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_38.npy',train_res)
np.save('val_38.npy',val_res)
#%%
x = base.layers[41].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_41.npy',train_res)
np.save('val_41.npy',val_res)
#%%
x = base.layers[44].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_44.npy',train_res)
np.save('val_44.npy',val_res)
#%%
x = base.layers[49].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_49.npy',train_res)
np.save('val_49.npy',val_res)
#%%
x = base.layers[50].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_50.npy',train_res)
np.save('val_50.npy',val_res)
#%%
x = base.layers[53].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_53.npy',train_res)
np.save('val_53.npy',val_res)

#%%
x = base.layers[56].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_56.npy',train_res)
np.save('val_56.npy',val_res)
#%%
x = base.layers[59].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_59.npy',train_res)
np.save('val_59.npy',val_res)
#%%
x = base.layers[60].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_60.npy',train_res)
np.save('val_60.npy',val_res)
#%%
x = base.layers[63].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_63.npy',train_res)
np.save('val_63.npy',val_res)
#%%
x = base.layers[65].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_65.npy',train_res)
np.save('val_65.npy',val_res)
#%%
x = base.layers[66].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_69.npy',train_res)
np.save('val_69.npy',val_res)
#%%
x = base.layers[70].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_70.npy',train_res)
np.save('val_70.npy',val_res)
#%%
x = base.layers[73].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_73.npy',train_res)
np.save('val_73.npy',val_res)

#%%
x = base.layers[76].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_76.npy',train_res)
np.save('val_76.npy',val_res)
#%%
x = base.layers[79].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_79.npy',train_res)
np.save('val_79.npy',val_res)
#%%
x = base.layers[80].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_80.npy',train_res)
np.save('val_80.npy',val_res)
#%%
x = base.layers[83].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_83.npy',train_res)
np.save('val_83.npy',val_res)
#%%
x = base.layers[86].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_86.npy',train_res)
np.save('val_86.npy',val_res)

#%%
x = base.layers[91].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_91.npy',train_res)
np.save('val_91.npy',val_res)
#%%
x = base.layers[92].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_92.npy',train_res)
np.save('val_92.npy',val_res)
#%%
x = base.layers[95].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_95.npy',train_res)
np.save('val_95.npy',val_res)
#%%
x = base.layers[98].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_98.npy',train_res)
np.save('val_98.npy',val_res)
#%%
x = base.layers[101].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_101.npy',train_res)
np.save('val_101.npy',val_res)
#%%
x = base.layers[102].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_102.npy',train_res)
np.save('val_102.npy',val_res)
#%%
x = base.layers[105].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_105.npy',train_res)
np.save('val_105.npy',val_res)
#%%
x = base.layers[108].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_108.npy',train_res)
np.save('val_108.npy',val_res)

x = base.layers[111].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_111.npy',train_res)
np.save('val_111.npy',val_res)
#%%
x = base.layers[112].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_112.npy',train_res)
np.save('val_112.npy',val_res)
#%%
x = base.layers[115].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_115.npy',train_res)
np.save('val_115.npy',val_res)
#%%
x = base.layers[118].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_118.npy',train_res)
np.save('val_118.npy',val_res)
#%%
x = base.layers[121].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_121.npy',train_res)
np.save('val_121.npy',val_res)
#%%
x = base.layers[122].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_122.npy',train_res)
np.save('val_122.npy',val_res)
#%%
x = base.layers[125].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_125.npy',train_res)
np.save('val_125.npy',val_res)

#%%
x = base.layers[128].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_128.npy',train_res)
np.save('val_128.npy',val_res)
#%%
x = base.layers[131].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_131.npy',train_res)
np.save('val_131.npy',val_res)
#%%
x = base.layers[132].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_132.npy',train_res)
np.save('val_132.npy',val_res)
#%%
x = base.layers[138].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_138.npy',train_res)
np.save('val_138.npy',val_res)
#%%
x = base.layers[141].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_141.npy',train_res)
np.save('val_141.npy',val_res)
#%%
x = base.layers[142].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_142.npy',train_res)
np.save('val_142.npy',val_res)
#%%
x = base.layers[145].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_145.npy',train_res)
np.save('val_145.npy',val_res)
#%%
x = base.layers[148].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_148.npy',train_res)
np.save('val_148.npy',val_res)

#%%
x = base.layers[153].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_153.npy',train_res)
np.save('val_153.npy',val_res)
#%%
x = base.layers[154].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_154.npy',train_res)
np.save('val_154.npy',val_res)
#%%
x = base.layers[157].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_157.npy',train_res)
np.save('val_157.npy',val_res)
#%%
x = base.layers[160].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_160.npy',train_res)
np.save('val_160.npy',val_res)
#%%
x = base.layers[163].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_163.npy',train_res)
np.save('val_163.npy',val_res)
#%%
x = base.layers[164].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_164.npy',train_res)
np.save('val_164.npy',val_res)
#%%
x = base.layers[167].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_167.npy',train_res)
np.save('val_167.npy',val_res)
#%%
x = base.layers[170].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_170.npy',train_res)
np.save('val_170.npy',val_res)
#%%
x = base.layers[173].output
x = Cropping2D(cropping=((8, 8), (8, 8)))(x)

model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_res = model.predict(train_x)
val_res = model.predict(val_x)
np.save('train_173.npy',train_res)
np.save('val_173.npy',val_res)



































