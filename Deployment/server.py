from fastapi import FastAPI, File, UploadFile
import uvicorn
from typing import List
import numpy as np
import glob as gb
import os
import shutil
from collections import Counter
import cv2
import keras
import tensorflow as tf

MODEL_PATH = './second_comeback_2_8.h5'
TEMP_FILE_PATH = 'temp/'
CODES = {'Babars':0, 'Bassem':1, 'Eman':2, 'El-Alfy':3, 'Nabiel':4, 'Nour':5, 'Ramadan':6, 'Roqia':7, 'Sabek':8, 'Foreign':9}

app = FastAPI()

# load the model
KerasModel = tf.keras.models.load_model(MODEL_PATH)

def remove(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the empty file
    elif os.path.isdir(path):
        shutil.rmtree(path) # remove file with its content

# get the highest probability of each list to get names then take the names of the highest 8 probabilities
def list_names(y_result , X_pred):
    #get the user name
    final_name=[]
    codes = []
    
    for i in range(len(X_pred)):
        codes.append((getcode(np.argmax(y_result[i])), y_result[i][np.argmax(y_result[i])]))
        # final_name.append(getcode(np.argmax(y_result[i])))
    return sorted(codes, key=lambda tup: tup[1], reverse=True)[:8]

# map predictive index to its corresponding name
def getcode(n) : 
    for x , y in CODES.items() : 
        if n == y : 
            return x

@app.get('/')
def index():
    return ('Welcome to the Face Recognition')

@app.post('/predict')
# pass the image to the prediction function to get their indexes
async def predict(data: List[UploadFile] = File(...)):
    size=140
    X_pred = []
    
    remove(TEMP_FILE_PATH)
    os.mkdir(TEMP_FILE_PATH)
    
    for file in data:
        with open(f'{TEMP_FILE_PATH}{file.filename}', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    img_files = gb.glob(pathname= str(TEMP_FILE_PATH+'*.jpg'))

    for img_file in img_files: 
        image = cv2.imread(img_file)
        # remove image less than 140 pix
        if image.shape[0]<140 and image.shape[1]<140:
            os.remove(img_file)  
        # resize the remainig images to 140 pix    
        else:    
            image_array = cv2.resize(image , (size,size))
            X_pred.append(list(image_array)) 

    # use the model to predict the user    
    X_pred_array = np.array(X_pred)
    y_result = KerasModel.predict(tf.cast(X_pred_array, tf.float32))
    remove(TEMP_FILE_PATH)
    # get the real user name as the most frequent in 8 list of names   
    final_name=list(zip(*list_names(y_result , X_pred)))[0]
    data = Counter(final_name)

    return max(final_name, key=data.get)


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1',port=8000)
