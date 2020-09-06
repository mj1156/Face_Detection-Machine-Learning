'''
@author: Abhishek Murali and Mukul Jain
'''
#%%
import os
# To change working directory in order to run the program 
#os.chdir('Insert the path')
os.chdir('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program')
#%%
import cv2
from joblib import load
import numpy as np
from skimage.feature import hog
encode_labels = {
  "A": "Angela Merkel",
  "D": "'Donald Trump",
  "H": "Horst Seehofer",
  "I": "Ingo Zamperoni",
  "K": "Karsten Schwanke",
  "M": "Martin Schulz",
  "U": "Unknown",
}

#loading the SVM model
load_clf= load('SVM_face_classification_model.joblib')

Image_HOG_features= np.zeros((1,648),dtype=np.float32())

count = 0
CasFile_face = cv2.CascadeClassifier('C:\Mukul\Documents\WS201920\ippr\Project 2020\haarcascade_frontalface_alt2.xml')
#CasFile_face = cv2.CascadeClassifier('C:/Users/CLOAK AND DAGGER/Downloads/profileFace10/haarcascade_profileface.xml')
# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('C:/Mukul/Documents/WS201920/ippr/Project 2020/Project Final/Test Video Segmentation/Test.mp4')
dim = (75,75)

ret = True
frc = 0  
while(ret == True):
    ret,frame = cap.read()
    if ret == False:
        break
    frc = frc+1
    # Load our image then convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = CasFile_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10,10))
    
    if faces is ():
        pass
    else:
        for x,y,w,h in faces:
            count = count + 1     
            c_img = gray[y:y+h, x:x+w]
            
            c_img = cv2.resize(c_img,dim)
            fin = cv2.resize(c_img,dim)
            
            fd = hog(fin, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1),block_norm='L2-Hys',visualize=False, multichannel=False)
            Image_HOG_features[0,:]=fd
            y_pre =load_clf.predict(Image_HOG_features)
            y_pre_prob =load_clf.predict_proba(Image_HOG_features)
            y_pred_max=np.amax(y_pre_prob[0])
            
            cv2.rectangle(frame, (x,y),(x+w,y+h),1, 2)
            y1 = y-15
            if (y - 15 )< 15 :
                y1=y + 15
                
            if (y_pred_max>0.85) :
                percentage =str(round(y_pred_max,3))    
                if y_pre == 'A':    
                    l=encode_labels['A']
                if y_pre == 'D':    
                    l=encode_labels['D']
                if y_pre == 'H':    
                    l=encode_labels['H']
                if y_pre == 'I':    
                    l=encode_labels['I']
                if y_pre == 'K':    
                    l=encode_labels['K']
                if y_pre == 'M':    
                    l=encode_labels['M']  
            else:
                percentage =str(round((1-y_pred_max),3))
                l=encode_labels['U']
        
            cv2.putText(frame,l,(x,y1),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,255,255),lineType=cv2.LINE_AA)
            cv2.putText(frame,percentage,(x+w,y1+h),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0),lineType=cv2.LINE_AA)
            
    cv2.imshow("Frame", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows() 
#%%
cv2.destroyAllWindows() 

#%%
import cv2
from joblib import load
import numpy as np
from skimage.feature import hog
encode_labels = {
  "A": "Angela Merkel",
  "D": "'Donald Trump",
  "H": "Horst Seehofer",
  "I": "Ingo Zamperoni",
  "K": "Karsten Schwanke",
  "M": "Martin Schulz",
  "U": "Unknown",
}

#loading the SVM model
load_clf= load('SVM_face_classification_model.joblib')

Image_HOG_features= np.zeros((1,648),dtype=np.float32())

count = 0
CasFile_face = cv2.CascadeClassifier('C:\Mukul\Documents\WS201920\ippr\Project 2020\haarcascade_frontalface_alt2.xml')
#CasFile_face = cv2.CascadeClassifier('C:/Users/CLOAK AND DAGGER/Downloads/profileFace10/haarcascade_profileface.xml')
# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('C:/Mukul/Documents/WS201920/ippr/Project 2020/Project Final/Test Video Segmentation/vidsegtest1.mp4')
size = (1280,720)
out = cv2.VideoWriter('E:/Project Final/Test Video Segmentation/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
dim = (75,75)

ret = True
frc = 0  
while(ret == True):
    ret,frame = cap.read()
    if ret == False:
        break
    frc = frc+1
    # Load our image then convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = CasFile_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10,10))
    
    if faces is ():
        pass
    else:
        for x,y,w,h in faces:
            count = count + 1     
            c_img = gray[y:y+h, x:x+w]
            
            c_img = cv2.resize(c_img,dim)
            fin = cv2.resize(c_img,dim)
            
            fd = hog(fin, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1),block_norm='L2-Hys',visualize=False, multichannel=False)
            Image_HOG_features[0,:]=fd
            y_pre =load_clf.predict(Image_HOG_features)
            y_pre_prob =load_clf.predict_proba(Image_HOG_features)
            y_pred_max=np.amax(y_pre_prob[0])
            
            cv2.rectangle(frame, (x,y),(x+w,y+h),1, 2)
            y1 = y-15
            if (y - 15 )< 15 :
                y1=y + 15
                
            if (y_pred_max>0.85) :
                percentage =str(round(y_pred_max,3))    
                if y_pre == 'A':    
                    l=encode_labels['A']
                if y_pre == 'D':    
                    l=encode_labels['D']
                if y_pre == 'H':    
                    l=encode_labels['H']
                if y_pre == 'I':    
                    l=encode_labels['I']
                if y_pre == 'K':    
                    l=encode_labels['K']
                if y_pre == 'M':    
                    l=encode_labels['M']  
            else:
                percentage =str(round((1-y_pred_max),3))
                l=encode_labels['U']
        
            cv2.putText(frame,l,(x,y1),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,255,255),lineType=cv2.LINE_AA)
            cv2.putText(frame,percentage,(x+w,y1+h),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0),lineType=cv2.LINE_AA)
            
    cv2.imshow("Frame", frame)
    out.write(frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
out.release()       
cv2.destroyAllWindows() 
#%%