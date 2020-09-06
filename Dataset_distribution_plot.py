
'''
@author: Abhishek Murali and Mukul Jain
'''
#%%
import os
# To change working directory in order to run the program 
#os.chdir('Insert the path')
os.chdir('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program')

#%%
import os
import matplotlib.pyplot as plt;
import numpy as np
import matplotlib.pyplot as plt

x_value =[]
y_value =[]
plt.show()
#Path="Insert the path " 
Path="C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\DataSet" 
# Extracting images
for o in os.listdir(Path):
    x_value.append(o)
total = np.arange(len(x_value))
for o in os.listdir(Path):
    y_value.append(len(os.listdir(os.path.join(Path, o))))

# plot     
plt.barh(total,y_value, align='center', alpha=0.5)
plt.yticks(total,x_value)
plt.xlabel('Number of Images')
plt.title('Training DataSet Distribution')

plt.show()    
#%%   