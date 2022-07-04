#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import glob as gb

from collections import Counter
import pandas as pd
import random
import cv2
import os
import keras
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
import tensorflow.keras
#import keras
from tensorflow import keras





# load the model 
KerasModel = tf.keras.models.load_model('second_comeback_2_8.h5')




#first take images and save them in the pre_test_cnn_input folder
def take_image():
    #use open cv to det the user images
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret , frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.8, minNeighbors=3)

        for face in faces:
            x, y, w, h = face

            roi_color = frame[y:y+h+15, x:x+w+15]
            #save the images in the pre_test_cnn_input folder
            file_name_path = "pre_test_cnn_input/"+"user"+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, roi_color)


            y_end_cord = y + h
            x_end_cord = x + w

            color=(255,0,0)
            strock=2
            cv2.rectangle(frame,(x,y),(x_end_cord,y_end_cord),color,strock)
            img_id+=1

        if ret==True:
             cv2.imshow('frame1',frame)
        
        if cv2.waitKey(1)==13 or int(img_id)==30:
                    break


    cap.release()
    cv2.destroyAllWindows()


#pass the image to the prediction function to get their indexes
def prediction(Model):

    size=140   
    take_image() 
    X_pred = []

    files = gb.glob(pathname= str('pre_test_cnn_input//*.jpg'))

    for file in files: 
        image = cv2.imread(file)
        #remove image less than 140 pix
        if image.shape[0]<140 and image.shape[1]<140:
            os.remove(file)  
        #resize the remainig images to 140 pix    
        else:    
            image_array = cv2.resize(image , (size,size))
            X_pred.append(list(image_array)) 

    #use the model to predict the user    
    X_pred_array = np.array(X_pred)
    y_result = Model.predict(tf.cast(X_pred_array,tf.float32))
    
    return y_result , X_pred

#set the ids
code = {'babars':0,'bassem':1,'eman':2,'elelfy':3,'nabiel':4,'nour':5,'ramadn':6,'roqia':7,'sabek':8,'not_valid':9}
#map predictive index to its corresponding name
def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x 
#get the highest probability of each list to get names then take the names of the highest 8 probabilities
def list_names():
    y_result , X_pred = prediction(KerasModel)  
    
    #get the user name
    final_name=[]
    codes = []
    
    for i in range(len(X_pred)):
        codes.append((getcode(np.argmax(y_result[i])), y_result[i][np.argmax(y_result[i])]))
        #final_name.append(getcode(np.argmax(y_result[i])))
    return sorted(codes, key=lambda tup: tup[1], reverse=True)[:8]

  
#get the real user name as the most frequent in 8 list of names   
def most_common():
    final_name=list(zip(*list_names()))[0]
    data = Counter(final_name)
    #print(final_name)
    return max(final_name, key=data.get)

the_user=most_common()
print(the_user)


# In[ ]:





# In[ ]:




