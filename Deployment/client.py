import requests
import cv2
import os
import shutil


URL = 'http://127.0.0.1:8000/predict'
TEMP_FILE_PATH = 'temp/'

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the empty file
    elif os.path.isdir(path):
        shutil.rmtree(path) # remove file with its content

# first take images and save them in the pre_test_cnn_input folder
def capture_images():
    # use open cv to det the user images
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    img_id = 0
    remove(TEMP_FILE_PATH)
    os.mkdir(TEMP_FILE_PATH)
    while True:
        ret , frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.8, minNeighbors=3)
        for face in faces:
            x, y, w, h = face

            roi_color = frame[y:y+h+15, x:x+w+15]
            # save the images in the pre_test_cnn_input folder
            file_name_path = TEMP_FILE_PATH +str(img_id)+ ".jpg"
            cv2.imwrite(file_name_path, roi_color)

            y_end_cord = y + h
            x_end_cord = x + w

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
   

if __name__ == "__main__":
    capture_images()
	
    files_list = []
    for root, dirs, files in os.walk(TEMP_FILE_PATH):
        for fileName in files:
            if len(files) > 0:
                files_list.append(('data', open(TEMP_FILE_PATH+fileName, 'rb')))

    response = requests.post(URL, files=files_list).text
    print(response)
    
        
