
'''
@author: Abhishek Murali and Mukul Jain
'''
#%%
import os
# To change working directory in order to run the program 
#os.chdir('Insert the path')
os.chdir('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program')
#%%
from joblib import load
import numpy as np
from matplotlib import image
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from os import listdir
i=0
j=0

load_clf= load('SVM_face_classification_model.joblib')
# folder='Insert the path of test dataset'
folder='C:/Mukul/Documents/WS201920/ippr/Project 2020/Project Final/Test Dataset/'
img_features = np.zeros((3371,648),dtype=np.float32())
labels = np.zeros((3371),dtype=np.str)
labels_proba = np.zeros((3371),dtype=np.str)
y_predicted_labels = np.zeros((3371),dtype=np.str)
file_name=listdir(folder)
image_list = list()

for files in file_name:
    #split the jpg and append with the numbers to list
    image_list.append(int(files.split('.jpg')[0]))
#sort the list to map features with labels
image_list.sort()
   
for i,filename in enumerate(image_list):
    img = image.imread(folder+str(' ')+str(filename)+'.jpg')
    fd = hog(img, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1),block_norm='L2-Hys',visualize=False, multichannel=False)
    img_features[i,:]=fd
    
# Predicting the labels from SVM model    
labels =load_clf.predict(img_features)
labels_proba =load_clf.predict_proba(img_features)

for j,k in enumerate(y_predicted_labels):
     if (np.amax(labels_proba[j])>0.80) :   
         y_predicted_labels[j]= labels[j] 
     else:
         y_predicted_labels[j]= 'U'
        
#Manual Labels imported from Labels.txt file
y_labels = np.zeros((3371),dtype=np.str)
f = open('Labels.txt', 'r')
for l,s in enumerate(f):
      y_labels[l]= s      
#Confusion Matrix 
conf_matrix=confusion_matrix(y_labels,y_predicted_labels)
print(conf_matrix)

print("Test Dataset score: {:.2f}".format(np.mean(y_labels == y_predicted_labels)))
#%%Confusion Matrix Plot
import matplotlib.pyplot as plt
i=0
j=0
class_names={'Angela Merkel','Donald Trump','Horst Seehofer','Ingo Zamperoni','Karsten Schwanke','Martin Schulz','Unknown'}
class_names_sorted=sorted(class_names)
fig, ax = plt.subplots()
im = ax.imshow(conf_matrix, cmap='Blues', interpolation='none')
cbar =  ax.figure.colorbar(im,ax=ax)
cbar.ax.set_ylabel('No of Images', rotation=-90, va="bottom")
# Plot non-normalized confusion matrix
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))

ax.set_xticklabels(class_names_sorted)
ax.set_yticklabels(class_names_sorted)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j,i, conf_matrix[i, j],ha="center", va="center", color="green")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
ax.set_title("Confusion Matrix")
fig.tight_layout()

plt.show()
#%%