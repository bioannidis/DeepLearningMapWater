
# coding: utf-8

# # Notes on Image Level Dataset Version 1

# In[89]:

# importing modules
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
url = 'http://umnlcc.cs.umn.edu/WaterDatasets/' + dataset_name + '.zip'
urllib.request.urlretrieve(url,dataset_name + '.zip')

zip_ref = zipfile.ZipFile( dataset_name + '.zip', 'r')
zip_ref.extractall()
zip_ref.close()
#os.system('unzip ' + dataset_name + '.zip')
print('Dataset Loaded ...')


# In[91]:

# dataset description
image_names = os.listdir(dataset_name)
print('* This dataset contains images and corresponding class labels of ' + str(len(image_names)) + ' lakes in .mat format.')
print('* Each .mat file has the following naming format:')
print('              data_<lake_id>.mat where lake_id is the ID of the lake (for example ' + image_names[0] + ')')
print('\n')
print('* Each .mat file has following variables: X, Y, qmap')
print('\n')
print('* X: A 3-d array of input features of size mxnxp where m is the number of rows, n is the number of columns and p is the number features.The value at (i,j,k) represents the feature value in kth band for the pixel at (i,j)')
print('\n')
print('* Y: A 2-d array of class labels of size mxn where m is the number of rows and n is the number of columns. The value at (i,j) represent the class label of pixel at (i,j). Class labels can take three possible values (1: water, 2: land, 3: unknown).')
print('\n')
print('* qmap: A 2-d array of data quality information of size mxn where m is the number of rows and n is the number of columns. The value at (i,j) represent quality flag of pixel at (i,j). Quality flag can take 9 possible values (1 to 9, 1 being highest quality and 9 being lowest quailty).')


# In[93]:

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

for b in range(X.shape[2]):
    X[:, :, b] = X[:, :, b] * 2.0 / (max_val - min_val)


plt.figure(figsize = (15,8))
plt.subplot(1,2,1)
plt.imshow(X[:,:,[0, 3, 2]])
plt.title('Color Composite Image')
plt.subplot(1,2,2)
plt.imshow(data['Y'])
plt.title('Class Label Map')
plt.show()


# In[ ]:

