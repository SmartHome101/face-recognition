#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import glob as gb
import keras
from collections import Counter
import pandas as pd
import random
import cv2
import os
import glob as gb
import tensorflow as tf
import keras
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
import tensorflow.keras
#import keras
from tensorflow import keras

#load the model
KerasModel = tf.keras.models.load_model('eight_members_ring.h5')


# In[3]:



#read the saved images and remove all images<140 pix then resize the remainig images        
size=180
X_pred = []
files = gb.glob(pathname= str('pre_test_cnn_input//*.jpg'))
for file in files: 
    
    image = cv2.imread(file)
    
    #remove image less than 140 pix
    if image.shape[0]<180 and image.shape[1]<180:
os.remove(file)  

    #resize the remainig images to 140 pix    
    else:    
image_array = cv2.resize(image , (size,size))
    
X_pred.append(list(image_array)) 




 
    

#use the model to predict how use the camera    
X_pred_array = np.array(X_pred)
#print(f'X_pred shape  is {X_pred_array.shape}')
y_result =KerasModel.predict(tf.cast(X_pred_array,tf.float32))
#KerasModel.predict(X_pred_array)
#print(y_result)




#set the ids

code = {'babers':0,'bassem':1,'elelfy':2,'eman':3,'ramadn':4,'roqia':5,'sabek':6,'salma':7}
def getcode(n) : 
    for x , y in code.items() : 
if n == y : 
    return x 





#get the user name
final_name=[]
for i in range(len(X_pred)):
   
  final_name.append(getcode(np.argmax(y_result[i])))
  #print(np.argmax(y_result[i]))
  #print(getcode(np.argmax(y_result[i])))
  

    
#print(final_name) 
    

    

    
    
    
#get the real user name    
def most_common(final_name):
    data = Counter(final_name)
    return max(final_name, key=data.get)

the_user=most_common(final_name)
#printr the user name
print(the_user)  

