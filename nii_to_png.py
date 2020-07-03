#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:00:06 2020

@author: alvi
"""

import os, nibabel
import imageio
def nii_to_png_3d(inputfile,outputfile,image_array):
  nx, ny, nz = image_array.shape
  # set destination folder
  if not os.path.exists(outputfile):
      os.makedirs(outputfile)
  total_slices = image_array.shape[2]
  slice_counter = 0
  # iterate through slices
  for current_slice in range(0, total_slices):
      # alternate slices
      if (slice_counter % 1) == 0:
          data = image_array[:, :, current_slice]
          #alternate slices and save as png
          if (slice_counter % 1) == 0:
              image_name = inputfile[:-4] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
              image_name=image_name.split("/")[-1]
              imageio.imwrite(outputfile+image_name, data)
              slice_counter += 1
def nii_to_png_4d(inputfile,outputfile,image_array):
  nx, ny, nz, nw = image_array.shape
  if not os.path.exists(outputfile):
      os.makedirs(outputfile)
      print("Created ouput directory: " + outputfile)
  print('Reading NIfTI file...')
  total_volumes = image_array.shape[3]
  total_slices = image_array.shape[2]
  # iterate through volumes
  for current_volume in range(0, total_volumes):
      slice_counter = 0
      # iterate through slices
      for current_slice in range(0, total_slices):
          if (slice_counter % 1) == 0:
              data = image_array[:, :, current_slice, current_volume]
              #alternate slices and save as png
              print('Saving image...')
              image_name = inputfile[:-4] + "_t" + "{:0>3}".format(str(current_volume+1)) + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
              image_name=image_name.split("/")[-1]
              imageio.imwrite(outputfile+image_name, data)
              slice_counter += 1

def nii_to_png(inputfile,outputfile):
    print('Input file is ', inputfile)
    print('Output folder is ', outputfile)
    # set fn as your 4d nifti file
    image_array = nibabel.load(inputfile).get_fdata()
    print(len(image_array.shape))
    # if 4D image inputted
    if len(image_array.shape) == 4:
        # set 4d array dimension values
        print("4D image")
        nii_to_png_4d(inputfile,outputfile,image_array)
        print('Finished converting images')
    # else if 3D image inputted
    elif len(image_array.shape) == 3:
        # set 4d array dimension values
        print("4D image")
        nii_to_png_3d(inputfile,outputfile,image_array)
        print('Finished converting images')
    else:
        print('Not a 3D or 4D Image. Please try again.')

source_path="/"
target_path="/"
for f in os.listdir(source_path):
  dr=f.split(".")[0]
  os.mkdir(target_path+dr)
  nii_to_png(inputfile = source_path+f,outputfile = target_path+dr+"/"+f)