
from scipy.io import matlab
import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
#get_ipython().magic('matplotlib inline')
import scipy.stats


# In[90]:

#loading data from the server. This will download the dataset in your curernt directory and then load it.
dataset_name = 'ImageLevelDataset_Version1'

image_names = os.listdir(dataset_name)


#display list of image names in the dataset
print(image_names)


# In[94]:

# Visualize a lake image
ID = '245045'
data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
X = data['X'].astype(np.float64)
min_val = 0
max_val = np.amax(X)
X[X[:, :, :] > max_val] = max_val
X[X[:, :, :] < min_val] = min_val

#for b in range(X.shape[2]):
#    X[:, :, b] = X[:, :, b] * 2.0 / (max_val - min_val)

s_patchsize1=int(X.shape[0]/10)
s_patchsize2=int(X.shape[1]/10)
s_patchind=0
Xpatches = np.zeros((s_patchsize1,s_patchsize2,X.shape[2],s_patchsize1*s_patchsize2))

for s_ind1 in range (s_patchsize1):
    for s_ind2 in range(s_patchsize2):
        Xpatches[:,:,:,s_patchind]=X[s_patchsize1*s_ind1:s_patchsize1*(s_ind1+1),s_patchsize2*s_ind2:s_patchsize2*(s_ind2+1),:]

plt.figure(figsize = (15,8))

plt.subplot(1,2,1)
plt.imshow(X[:,:,[0, 3, 2]])
plt.title('Color Composite Image')
plt.subplot(1,2,2)
plt.imshow(data['Y'])
plt.title('Class Label Map')
plt.show()


# In[ ]:

