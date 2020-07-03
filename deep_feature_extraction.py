#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:21:32 2020

@author: alvi
"""

import os,cv2,numpy as np,csv,warnings
from keras.optimizers import SGD
from keras.models import Model
from keras.applications import vgg19
from keras.layers.core import Dense

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
      #print(jpg_path[start+i])
      img=cv2.imread(jpg_path[start+i])

      im = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      ima[i]=im
  return ima

def extract_imagenet_features(data_path=None,features_path=None,weight_path='imagenet',output_layer='fc1',base_model=None,step_size=500,done=0):
    count_train=4000
    image_paths,label=gather_paths_all(data_path)
    count_train=len(label)
    X_train1 = image_paths[:]
    Y_train1=label[:]
    new_model=build_model(base_model=base_model,output_layer='fc1',num_classes=6)
    if(not weight_path=='imagenet'):
      new_model.load_weights(weight_path)
    
    layer_output = new_model.get_layer(output_layer).output
    x=layer_output
    model1 = Model(new_model.input, outputs=x)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss='mean_squared_error', optimizer=sgd)
    return extract_features(model=model1,layer=None,X_train1=X_train1,Y_train1=Y_train1,features_path=features_path,step_size=step_size,count=count_train,done=done)

def extract_features(model=None,layer=None,X_train1=None,Y_train1=None,features_path=None,step_size=500,count=5293,done=0):
  warnings.filterwarnings('ignore')
  output=[]
  steps=int(count/step_size)+1
  for i in range(done,steps):
    st=i*step_size
    end=(i+1)*step_size
    if(i==steps-1):
      end=count
    Y=Y_train1[st:end]
    images=gather_images_from_paths(X_train1,st,end-st,img_rows=224,img_cols=224)
    if(layer!=None):
        output=layer.predict(images)
    output=model.predict(images)
    ch='a'
    if(i==0):
      ch='w'
    with open(features_path, ch, newline='') as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
      for i in range(len(output)):
        spamwriter.writerow(np.concatenate((output[i],[Y[i]])))
  return

def build_model(base_model=None,output_layer='avg_pool',num_classes=8):
  layer_output = base_model.get_layer(output_layer).output
  for layer in base_model.layers[:-3]:
    layer.trainable = False
  x=layer_output
  x = Dense(1024, activation='sigmoid',name='1024_out')(x)
  x = Dense(512, activation='sigmoid', name='512_out')(x)
  x = Dense(num_classes, activation='sigmoid',name="final")(x)
  model = Model(base_model.input, outputs=x)
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=sgd)
  return model

base_model=vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#output_layer='final'
output_layer='512_out'
data_path="/"
weights_path="weights/weights_tb.h5"
features_path='features.csv'
extract_imagenet_features(data_path=data_path,features_path=features_path,weight_path=weights_path,output_layer=output_layer,base_model=base_model,step_size=1000,done=5)
