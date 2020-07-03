#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alvi
"""
import cv2, os, numpy as np, pandas as pd
from skimage import feature
import math

def gather_images_from_paths(jpg_path,start,count,img_rows,img_cols):
  ima=np.zeros((count,img_rows,img_cols,3))
  for i in range(count):
      img=cv2.imread(jpg_path[start+i])
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
                                                        
def lbp_feature(img,radius=1,eps=1e-7):
    numPoints=4*radius
    image=cv2.imread(img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray_image, numPoints,radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return lbp,hist

def lbp_features_each(data_path='/',storage_path='/',paths=None,labels=None,radius=1,done=0):
    points=radius*4
    step1=step=1000
    count=len(paths)
    parts=count/step
    count-=1
    for part in range(done,math.ceil(parts)):
      if(part==13):
        return
      print(part)
      if(len(paths[part*step:])<step):
        step1=len(paths[part*step:])
      f=np.zeros((step1,points+2),float)
      for i in range(step1):
        (_,f[i])=lbp_feature(paths[(part*step)+i],radius=radius)
        count-=1
        if(count<0):
          break
      df=pd.DataFrame(f)
      df = df.assign(Lbl=labels[(part*step):(part*step)+step1])
      df.to_csv(storage_path+"_"+str(part)+".csv")
    return df

data_path="/"
paths_train,labels_train=gather_paths_all(data_path)
r=0
feature_path="/"
lbp_features_each(data_path=data_path,storage_path=feature_path,paths=paths_train,labels=labels_train,radius=r+1,done=0)

