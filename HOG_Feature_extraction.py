
'''
@author: Abhishek Murali and Mukul Jain
'''
#%%
import os
# To change working directory in order to run the program 
#os.chdir('Insert the path')
os.chdir('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program')

#%%

import numpy as np
import random
from skimage.feature import hog
from matplotlib import image
from os import listdir

#path = 'Insert the path for the training DataSet'
path ='C:/Mukul/Documents/WS201920/ippr/Project 2020/Project Final/Training_DataSet/'

#Create a 2D array ( no of elements X no of features) for  HOG features
feature = np.zeros((6637,648),dtype=np.float32())
labels = np.zeros((6637),dtype=np.str)
file_list=listdir(path)

# To  introduce the randomness in the training dataset
random.shuffle(file_list)
for i,filename in enumerate(file_list):
    img = image.imread(path+file_list[i])
    fd = hog(img, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1),block_norm='L2-Hys',visualize=False, multichannel=False)
    feature[i,:]=fd
    labels[i] = filename[0]
  
# features and labels as numpy array for training     
np.save('feature.npy',feature)
np.save('labels.npy',labels)
#%%
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.feature import hog
from matplotlib import image
from skimage import exposure
from skimage import filters
from skimage.filters import meijering, sato, frangi, hessian

import cv2
import numpy as np
img = image.imread('C:\Mukul\Master Thesis\image2.png',0)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(gray.shape)
#laplacian = cv2.Laplacian(gray,cv2.CV_16S)
#plt.figure(figsize =(10,10))
#plt.hist(img.ravel(),256,[0,256])
#plt.show()


img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
kernel1 = np.ones((5,5),np.uint8)
#gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel1)
#closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)

blur1 = cv2.GaussianBlur(img,(5,5),0)
laplacian = cv2.Laplacian(blur1,cv2.CV_64F)
#result =  hessian(img)
#can = cv2.Canny(img,0,200)
#cv2.imshow('imageb1',can)
#fd,image1 = hog(img, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1),block_norm='L2-Hys',visualize=True, multichannel=True)
#edge_sobel = filters.sobel(image1)
#image_rescaled = exposure.rescale_intensity(image1, in_range=(0, 10))
#cv2.imshow('imageb1',gray)

plt.figure(figsize = (10,10))
plt.imshow(laplacian,cmap='gray')
plt.title('Histogram of Oriented Gradients')
plt.show()
cv2.waitKey(0)
#%%
import cv2
import numpy as np
CasFile_face = cv2.CascadeClassifier('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program\haarcascade_frontalface_alt2.xml')
#img = image.imread('C:\Mukul\DSLR PICS\Herdentor Trip\IMG_1977.jpg',0)
img = image.imread('C:\Mukul\Master Thesis\image1.png',0)
faces = CasFile_face.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10, minSize=(10,10))
dim = (250,250) 
count =0   
if faces is ():
    print("no faces")

else:
    for x,y,w,h in faces:
        count = count + 1     
        c_img = img[y:y+h, x:x+w]
        c_img = cv2.resize(c_img,dim)
        fin = cv2.resize(c_img,dim)
            #cv2.imwrite('Insert the path'+str(count)+'.jpg',fin)
        cv2.imwrite(r'C:\Users\CLOAK AND DAGGER\Downloads\ '+str(count)+'.jpg',fin)


#%%
import cv2
cv2.destroyAllWindows()
#%%