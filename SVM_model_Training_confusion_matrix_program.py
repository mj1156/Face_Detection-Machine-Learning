'''
@author: Abhishek Murali and Mukul Jain
'''
#%%
import os
# To change working directory in order to run the program 
#os.chdir('Insert the path')
os.chdir('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program')

#%%
# Training dataset is distributed into train and test dataset for SVM training 
import numpy as np
from sklearn import svm
from joblib import dump
from sklearn.metrics import confusion_matrix

per_t = np.int_(6637*0.8)
per_tv = np.int_(6637*0.2)
x_train = np.zeros((per_t,648),dtype=np.float32())
x_test = np.zeros((per_tv+1,648),dtype=np.float32())
y_train = np.zeros((per_t),dtype=np.str)
y_test = np.zeros((per_tv+1),dtype=np.str)
i,j,k = 0,0,0 
feature = np.load('feature.npy')
labels = np.load('labels.npy')

# Used for creating train and test dataset
for i,el in enumerate(feature):
    if (i<=per_t-1):
        x_train[i,:]=el
        y_train[i]=labels[i]
    else:
        x_test[j,:]=el
        y_test[j]=labels[i]
        j=j+1

#SVM Training
clf = svm.SVC(C = 0.1,kernel = 'poly',gamma = 'scale',probability=True)
clf.fit(x_train,y_train)
y_res = clf.predict(x_test)

# Saving the SVM model
dump(clf, 'SVM_face_classification_model.joblib')

# Confusion Matrix
cm=confusion_matrix(y_test,y_res)
print(cm)



#%% Confusion Matrix plot
import matplotlib.pyplot as plt
class_names={'Angela Merkel','Donald Trump','Horst Seehofer','Ingo Zamperoni','Karsten Schwanke','Martin Schulz'}
class_names_sorted=sorted(class_names)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues', interpolation='none')
cbar =  ax.figure.colorbar(im,ax=ax)
cbar.ax.set_ylabel('No of Images', rotation=-90, va="bottom")

# Plot non-normalized confusion matrix
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))

ax.set_xticklabels(class_names_sorted)
ax.set_yticklabels(class_names_sorted)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j,i, cm[i, j],
                       ha="center", va="center", color="w")
        
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
ax.set_title("Confusion Matrix")
fig.tight_layout()

plt.show()


#%%
