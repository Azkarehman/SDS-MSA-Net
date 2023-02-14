# -*- coding: utf-8 -*-
"""train

"""

import nibabel as nib
import numpy as np
import cv2
import os
from glob import glob
from skimage.transform import resize
from skimage import morphology, measure
from sklearn.cluster import KMeans
import scipy.ndimage
from scipy.ndimage import label
import matplotlib.pyplot as plt

"""# Model_Training

## Data generator
"""

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs,to_fit=True, batch_size=32, dim=(8 ,512, 512), shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data

        if self.to_fit:
            X,y = self._generate_Xy(list_IDs_temp)
            return X, y
        else:
            X = self._generate_X(list_IDs_temp)
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_Xy(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X1 = np.empty((self.batch_size,5,160,160,1))
        X2 = np.empty((self.batch_size,160,160,1))
        y1 = np.empty((self.batch_size,160,160,1))
        y2 = np.empty((self.batch_size,160,160,2))
        y3 = np.empty((self.batch_size,160,160,3))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img1 = np.load(ID)
            mas1 = np.load(ID.replace('Inps','Outs'))
            X1[i,:,:,:,0] = np.moveaxis(img1,-1,0)
            X2[i,:,:,0] = img1[:,:,2]
            y1[i,:,:,0] = mas1[:,:,0]
            y2[i,:,:,0] = mas1[:,:,0]
            y3[i,:,:,0] = mas1[:,:,0]
            y2[i,:,:,1] = mas1[:,:,1]
            y3[i,:,:,1] = mas1[:,:,1]
            y3[i,:,:,2] = mas1[:,:,2]
        return [X1,X2],[y1,y2,y3,y3]

a = glob('/content/Preprocessed/Inps/*')

train_datagen = DataGenerator(list_IDs=a,to_fit=True, batch_size=4, dim=(512, 512), shuffle=True)

x,y = train_datagen.__getitem__(1)


"""## Defining Model and Training"""

import tensorflow as tf
from tensorflow.keras import backend as K

def Res_block(x1,filter):
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x1)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Layer 3
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Layer 2_1
    x2 = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x1)
    x2 = tf.keras.layers.Activation('relu')(x2)
    x2 = tf.keras.layers.BatchNormalization(axis=3)(x2)
    x = tf.keras.layers.Add()([x, x2])
    return x

def Bottleneck_block(x1,filter,size1 = 5,size2 = 2):
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x1)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(size1,size2,size2), padding='same')(x)
    print(int(x.shape[2]),int(x.shape[3]),int(x.shape[4]))
    x = tf.keras.layers.Reshape((int(x.shape[2]),int(x.shape[3]),int(x.shape[4])))(x)
    return(x)

def encoder_3d(x_input):
    X = Res_block(x_input,8)
    X0 = Bottleneck_block(X,8,size1 = 5,size2 = 2)
    X = Res_block(X,16)
    X1 = Bottleneck_block(X,16,size1 = 5,size2 = 4)
    X = Res_block(X,32)
    X2 = Bottleneck_block(X,32,size1 = 5,size2 = 8)
    X = Res_block(X,64)
    X3 = Bottleneck_block(X,64,size1 = 5,size2 = 16)
    return(X0,X1,X2,X3)

def Conv_block(x,filter):
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    return x

def encoder_2nd(X_input,X0,X1,X2,X3):
    Y1 = Conv_block(X_input,8)
    Y_1 = tf.keras.layers.Concatenate()([Y1,X0])
    Y2 = Conv_block(Y_1,16)
    Y_2 = tf.keras.layers.Concatenate()([Y2,X1])
    Y3 = Conv_block(Y_2,16)
    Y_3 = tf.keras.layers.Concatenate()([Y3,X2])
    Y4 = Conv_block(Y_3,16)
    Y_4 = tf.keras.layers.Concatenate()([Y4,X3])
    return(Y_1,Y_2,Y_3,Y_4)

def Attention_block(x1,g1,filter):
    x1 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x1)
    g1 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(g1)
    x = tf.keras.layers.Add()([x1, g1])     
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Multiply()([x1, x])
    return x

def DeConv_block(x,filter):
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D(size=2)(x)
    return x

def Auxilary_block(x,filter):
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.Activation('softmax')(x)
    return(x)

def main_decoder(Y_1,Y_2,Y_3,Y_4):
    Z_3 = DeConv_block(Y_4,32)
    Z = Attention_block(Z_3,Y_3,32)
    Z_2 = DeConv_block(Z,16)
    Z = Attention_block(Z_2,Y_2,32)
    Z_1 = DeConv_block(Z,8)
    Z = Attention_block(Z_1,Y_1,32)
    Z = DeConv_block(Z,3)
    Z = tf.keras.layers.Conv2D(3, (3,3), padding = 'same')(Z)
    Z = tf.keras.layers.Activation('softmax')(Z)    

    Z_3 = tf.keras.layers.UpSampling2D(size=8)(Z_3)
    Z_3 = Auxilary_block(Z_3,1)
    Z_2 = tf.keras.layers.UpSampling2D(size=4)(Z_2)
    Z_2 = Auxilary_block(Z_3,2)
    Z_1 = tf.keras.layers.UpSampling2D(size=2)(Z_1)
    Z_1 = Auxilary_block(Z_1,3)
    return(Z_3,Z_2,Z_1,Z)

x_input1 = tf.keras.layers.Input((5,160,160,1))
x_input2 = tf.keras.layers.Input((160,160,1))
X0,X1,X2,X3 = encoder_3d(x_input1)
X_1,X_2,X_3,X_4 = encoder_2nd(x_input2,X0,X1,X2,X3)
Z_3,Z_2,Z_1,Z = main_decoder(X_1,X_2,X_3,X_4)
model = tf.keras.models.Model(inputs = [x_input1,x_input2], outputs = [Z_3,Z_2,Z_1,Z], name = "ResNet34")
print(model.summary())

Opt= tf.keras.optimizers.SGD(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)

model.compile(optimizer=Opt, loss='binary_crossentropy')

hist = model.fit(train_datagen,epochs=600)

