# -*- coding: utf-8 -*-
"""preprocess
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

def get_img_mask(ind):
    nii_data  = nib.load('BRATS_2020/BraTS20_Training_'+str(ind)+'/BraTS20_Training_'+str(ind)+'_flair.nii.gz')
    nii_img = nii_data.get_fdata()
    nii_data  = nib.load('BRATS_2020/BraTS20_Training_'+str(ind)+'/BraTS20_Training_'+str(ind)+'_seg.nii.gz')
    nii_mas = nii_data.get_fdata()
    return(nii_img,nii_mas)

def preprocess_scan(img,mask):
    img_temp = np.amax(img,axis=-1)
    a,b = np.where(img_temp>0)
    a = int(np.average(a))
    b = int(np.average(b))
    A1 = a-80
    A2 = a+80
    B1 = b-80
    B2 = b+80
    img = img[A1:A2,B1:B2,:]
    mask = mask[A1:A2,B1:B2,:]
    for i in range(img.shape[2]):
      temp = img[:,:,i]
      if(np.sum(temp)>0):
        temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
        temp = temp*255
        temp = temp.astype(np.uint8)
        equ = cv2.equalizeHist(temp)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
        img[:,:,i] = clahe.apply(temp)
    mask2 = np.zeros((160,160,mask.shape[2],3))        
    for i in range(img.shape[2]):
      Mas = mask[:,:,i]
      # Biggest
      mas = Mas.copy()
      mas[mas==1]=2
      mas[mas==4]=2
      mas[mas>0]=1
      mask2[:,:,i,0] = mas
      # Medium
      mas = Mas.copy()
      mas[mas==2]=0
      mas[mas==1]=4
      mas[mas>0]=1
      mask2[:,:,i,1] = mas
      # Smallest
      mas = Mas.copy()
      mas[mas==2]=0
      mas[mas==4]=0
      mas[mas>0]=1
      mask2[:,:,i,2] = mas
    return(img,mask2)

def save_preprocess_slices(img,mask,name):
  a = np.unique(np.where(mask[:,:,:,2]>0)[2])
  if not os.path.exists('Preprocessed'):
      os.makedirs('Preprocessed')
  if not os.path.exists('Preprocessed/Inps'):
      os.makedirs('Preprocessed/Inps')
  if not os.path.exists('Preprocessed/Outs'):
      os.makedirs('Preprocessed/Outs')
  for i in (a):
      np.save('Preprocessed/Inps/'+str(name)+'_'+str(i),img[:,:,i-3:i+2])
      np.save('Preprocessed/Outs/'+str(name)+'_'+str(i),mask[:,:,i,:])
    
    
nii_img,nii_mas = get_img_mask('019')
nii_img,nii_mas = preprocess_scan(nii_img,nii_mas)

plt.imshow(nii_img[:,:,55])
plt.show()
plt.imshow(nii_mas[:,:,55,0])
plt.show()
plt.imshow(nii_mas[:,:,55,1])
plt.show()
plt.imshow(nii_mas[:,:,55,2])
plt.show()

save_preprocess_slices(nii_img,nii_mas,'19')
