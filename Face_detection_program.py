
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

count =0
# Loading Cascade Classifier file for detection 
#CasFile_face = cv2.CascadeClassifier('Insert the path')
CasFile_face = cv2.CascadeClassifier('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program\haarcascade_frontalface_alt2.xml')

# Create a VideoCapture object and read from input file 
#cap = cv2.VideoCapture('Insert the path')
cap = cv2.VideoCapture('C:/Mukul/Documents/WS201920/ippr/Project 2020/Project Final/Test Video Segmentation/vidsegtest5.mp4')
dim = (75,75)
ret = True
frc = 0  
while(ret == True):
    ret,frame = cap.read()
    if ret == False:
        break
    frc = frc+1
    cv2.imshow('image',frame)
    cv2.waitKey(5)
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
            #cv2.imwrite('Insert the path'+str(count)+'.jpg',fin)
            cv2.imwrite('C:\Mukul\Documents\WS201920\ippr\Project 2020\Test Videos\Test\ '+str(count)+'.jpg',fin)
            
cv2.destroyAllWindows()           

#%%
cv2.destroyAllWindows() 
#%%