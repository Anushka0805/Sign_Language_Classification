# -*- coding: utf-8 -*-

import os, sys
import cv2
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import numpy as np
import pandas as pd
from subprocess import check_output
from scipy import ndimage
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import normalize, to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#to convert image data into np array called Dataset and Labels
def imgtoarr(inpath, outpath):
  folder=outpath
  
  if not os.path.exists(folder):
          os.mkdir(folder)
          print("Directory " , folder ,  " Created ")
  else:    
          print("Directory " , folder ,  " already exists")

    #to move all images into one single folder
  k=0
  for j in ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','_']:
    for i in range (1,1501):
      os.rename(inpath + str(j) +'/'+str(i)+ '.jpg', outpath+'/' + str(k)+'_'+str(i)+ '.jpg')
    k=k+1


  onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

  train_files = []
  y_train = []
  i=0
  for _file in onlyfiles:
      train_files.append(_file)
      
  print("Total number of images: %d" % len(train_files))

  # Dimensions
  image_width = 50
  image_height = 50

  channels = 1
  nb_classes = 37

  dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                      dtype=np.float32)
  labels=np.ndarray(shape=(len(train_files),1),
                      dtype=np.int64)
  i = 0
  for _file in train_files:
      img = load_img(folder + "/" + _file, color_mode='grayscale')  # this is a PIL image
      img.thumbnail((image_width, image_height))
      # Convert to Numpy Array
      x = img_to_array(img)  
      dataset[i] = x
      label_in_file = _file.find("_")
      labels[i]=(int(_file[0:label_in_file]))
      i += 1
      
  print("All images have been converted to arrays!")
  return (dataset, labels)
#to convert to one hot encoding
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def split_data(inpath, outpath):
  dataset, labels=imgtoarr(inpath, outpath)
#Splitting into train, val and test 
  x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=33)
  x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=33)
  print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(x_train), len(x_val), len(x_test)))

  y_train_orig=y_train.T
  y_test_orig=y_test.T
  y_val_orig=y_val.T

  X_train = x_train/255.
  X_test = x_test/255.
  X_val = x_val/255.
#one hot encoding
  Y_train = convert_to_one_hot(y_train_orig, 37).T
  Y_test = convert_to_one_hot(y_test_orig, 37).T
  Y_val = convert_to_one_hot(y_val_orig, 37).T

  return X_train,X_test,X_val,Y_train,Y_test,Y_val


#making prediction on image data
def predict_img_data(path, model):
  img = cv2.imread(path)
  img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  #masking
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_blue = np.array([0, 0, 120])
  upper_blue = np.array([180, 38, 255])
  mask = cv2.inRange(hsv, lower_blue, upper_blue)
  result = cv2.bitwise_and(img, img, mask=mask)
  b, g, r = cv2.split(result)  
  filter = g.copy()
  ret,mask = cv2.threshold(filter,10,255, 1)

  #resizing
  resized = cv2.resize(mask, (50,50))

#convert to array
  test = img_to_array(resized)

  eg = np.ndarray(shape=(1, 50, 50, 1),
                     dtype=np.float32)
  eg[0] = test

  eg2=eg/255.

  Xnew=eg2[:1]
  ynew=np.argmax(model.predict(Xnew), axis=-1)
  
  # matching labels to corresponding classes
  if (ynew==10):
      ynew='A'
  elif (ynew==11):
      ynew='B'
  elif (ynew==12):
      ynew='C'
  elif (ynew==13):
      ynew='D'
  elif (ynew==14):
      ynew='E'
  elif (ynew==15):
      ynew='F'
  elif (ynew==16):
      ynew='G'
  elif (ynew==17):
      ynew='H'
  elif (ynew==18):
      ynew='I'
  elif (ynew==19):
      ynew='J'
  elif (ynew==20):
      ynew='K'
  elif (ynew==21):
      ynew='L'
  elif (ynew==22):
      ynew='M'
  elif (ynew==23):
      ynew='N'
  elif (ynew==24):
      ynew='O'
  elif (ynew==25):
      ynew='P'
  elif (ynew==26):
      ynew='Q'
  elif (ynew==27):
      ynew='R'
  elif (ynew==28):
      ynew='S'
  elif (ynew==29):
      ynew='T'
  elif (ynew==30):
      ynew='U'
  elif (ynew==31):
      ynew='V'
  elif (ynew==32):
      ynew='W'
  elif (ynew==33):
      ynew='X'
  elif (ynew==34):
      ynew='Y'
  elif (ynew==35):
      ynew='Z'
  elif (ynew==36):
      ynew='_'

  
  print("input image")
  pyplot.imshow(img)
  pyplot.show
  
  print("resized mask of the input image")
  pyplot.imshow(resized)
  pyplot.show
  
  print("Prediction=%s" % (ynew))
  