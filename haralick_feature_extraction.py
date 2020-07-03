#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alvi
"""

import cv2, numpy as np
import os
#!pip3 install mahotas
import mahotas as mh
import pandas as pd

def gather_images_from_paths(jpg_path,start,count,img_rows,img_cols):
  ima=np.zeros((count,img_rows,img_cols,3))
  for i in range(count):
      img=cv2.imread(jpg_path[start+i])
      print(jpg_path[start+i])
      im = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      ima[i]=im
  return ima

def gather_paths_all(jpg_path):
  count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)])
  folder=os.listdir(jpg_path)
  ima=['' for x in range(count)]
  label=[0 for x in range(count)]
  for fldr in folder:
      temp=[f for f in os.listdir(jpg_path+fldr+"/") if f.endswith(".png")]
      for f in temp:
          im=jpg_path+fldr+"/"+f
          count-=1
          ima[count]=im
          label[count]=int(fldr.split("_")[-1])
  return ima[count:],label[count:]

def extract_haralick_features(X,Y,haralick_features_path,counttest=0,img_rows=224,img_cols=224,done=0):
  count=1000
  i=done
  while i<counttest:
    total=count
    if(i+count>=counttest):
      total=counttest-i
    ima=gather_images_from_paths(X,i,total,img_rows,img_cols)
    hf=[mh.features.haralick(im.astype(np.uint8)).ravel() for im in ima]
    df = pd.DataFrame(data=hf)
    df = df.assign(Lbl=Y[i:i+total])
    with open(haralick_features_path, 'a') as f:
      df.to_csv(f,index=False,header=False)
    i+=count
  return

data_path="/"
paths_img,labels=gather_paths_all(jpg_path=data_path)
count=len(labels)
features_path='features.csv'
extract_haralick_features(data_path,labels,features_path,counttest=count,img_rows=224,img_cols=224,done=0)