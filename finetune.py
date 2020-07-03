#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alvi
"""

import cv2, numpy as np, os, pandas as pd
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.utils import shuffle
#import warnings
#warnings.filterwarnings('ignore')
from tensorflow.python.keras.applications.vgg19 import VGG19

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

def gather_images_from_paths(jpg_path,start,count,img_rows=224,img_cols=224):
  ima=np.zeros((count,img_rows,img_cols,3))
  for i in range(count):
      img=cv2.imread(jpg_path[start+i])
      im = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      ima[i]=im
  return ima

def finetune_chunks(base_model=None,output_layer='avg_pool',weights='imagenet',include_top=True,step_size=1000,image_paths=None, labels=None,count_train=5293,weights_new=None,nb_epoch=1):
  model=build_model(base_model,output_layer)
  for i in range(int(count_train/step_size)+1):
    st=int((step_size)*i)
    end=int((step_size)*(i+1))
    if(i>=int(count_train/step_size)):
        end=count_train
    if(st>=end):
      break
    mid=int((st+end)/2)
    X_train1, Y_train1 = shuffle(image_paths, labels, random_state=i)
    X_train=gather_images_from_paths(X_train1[st:mid],start=0,count=mid-st)
    X_test=gather_images_from_paths(X_train1[mid:end],start=0,count=end-mid)
    Y_train=Y_train1[st:mid]
    Y_test=Y_train1[mid:end]
    print(i,st,mid,end,X_train.shape,Y_train.shape,Y_train.shape,Y_test.shape)
    # Start Fine-tuning
    checkpoint = ModelCheckpoint(weights_new+"_ic20_checkpoint_"+str(i)+".h5", monitor='loss', verbose=1,save_best_only=True, mode='auto', period=1)
    model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,shuffle=True,verbose=1,validation_data=(X_test, Y_test),callbacks=[checkpoint])
  weights=weights_new+"_ic20_512.h5"
  model.save(weights)
  return

def build_model(base_model=None,output_layer='avg_pool'):
  layer_output = base_model.get_layer(output_layer).output
  for layer in base_model.layers[:-3]:
    layer.trainable = False
  x=layer_output
  x = Dense(1024, activation='sigmoid')(x)
  x = Dense(512, activation='sigmoid')(x)
  x = Dense(num_classes, activation='sigmoid')(x)
  model = Model(base_model.input, outputs=x)
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=sgd)
  return model

def label_map(label,gtf):
  labels=pd.DataFrame()
  labels["Filename"]=label
  gt=pd.read_csv(gtf)
  gt['Filename'] = gt['Filename'].str[8:11]
  gt['Filename'] = gt['Filename'].astype(int)
  return pd.merge(labels, gt, how='outer', left_on="Filename", right_on='Filename').drop("Filename",axis=1)

image_path="/"
weight_path="/"
gtf="ground_truths.csv"
img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 6

test_size=0.70
batch_size = 30

image_paths,label=gather_paths_all(image_path)
labels=label_map(label,gtf)

count_train=len(labels)
nb_epoch = 50
step_size=4000

base_model=VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
finetune_chunks(base_model=base_model,output_layer='fc1',weights='imagenet',include_top=True,step_size=step_size,image_paths=image_paths, labels=labels,count_train=count_train,weights_new=weight_path,nb_epoch=nb_epoch)


