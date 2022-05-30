#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob as gb
import cv2
import os


#use open cv to det the user images
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
img_id = 0


#files = gb.glob(pathname= str('pre_test_cnn_input//*.jpg'))
#for file in files:
 #    os.remove(file) 

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.8, minNeighbors=3)
  
    for face in faces:
        x,y,w,h=face
        
        
        roi_color = frame[y:y+h+40, x:x+w+40]  #(ycord_start, ycord_end)
        
        
        
        #save the frame images
        #change the name before run again
        file_name_path = "pre_test_cnn_input/"+"user"+str(img_id)+".jpg"
        cv2.imwrite(file_name_path, roi_color)
           
        
        y_end_cord=y+h
        x_end_cord=x+w
        color=(255,0,0)
        strock=2
        cv2.rectangle(frame,(x,y),(x_end_cord,y_end_cord),color,strock)
        img_id+=1
           
   

    if ret==True:
         cv2.imshow('frame1',frame)
    if cv2.waitKey(1)==13 or int(img_id)==20:
                break

        
cap.release()
cv2.destroyAllWindows()
        

