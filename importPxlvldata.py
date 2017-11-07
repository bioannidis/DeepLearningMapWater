
# coding: utf-8

# In[11]:


# Notes on Pixel Level Dataset Version 1


# In[1]:


# importing modules and data (execute if using python 2.x)
# from scipy.io import matlab
# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import urllib
# from sklearn import svm
# get_ipython().magic(u'matplotlib inline')
#
# #loading data from the server. This will download the dataset in your curernt directory and then load it.
# url = 'http://umnlcc.cs.umn.edu/WaterDatasets/PixelLevelDataset_Version1.mat'
# urllib.urlretrieve(url,'PixelLevelDataset_Version1.mat')
# data = matlab.loadmat('PixelLevelDataset_Version1.mat')
# print('Dataset Loaded ...')


# In[2]:


# importing modules and data (execute if using python 3.x)
# importing modules
from scipy.io import matlab
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tensorflow as tf
from sklearn import svm
# get_ipython().magic(u'matplotlib inline')

#loading data from the server. This will download the dataset in your curernt directory and then load it.
url = 'http://umnlcc.cs.umn.edu/WaterDatasets/PixelLevelDataset_Version1.mat'
urllib.request.urlretrieve(url,'PixelLevelDataset_Version1.mat')
data = matlab.loadmat('PixelLevelDataset_Version1.mat')
print('Dataset Loaded ...')


# In[3]:


# dataset description
print('This dataset has following variables: X_all, Y_all, QUALITY_all, ID_all')
print('\n')
print('X_all: A 2-d array of input features of size ' + str(data['X_all'].shape) + ' where each row corresponds to 7 feature values of a pixel.')
print('\n')
print('Y_all: A 1-d array of size ' + str(data['Y_all'].shape) + ' containing class labels (1: water, 2: land, 3: unknown). Ignore the instances with uknown label during training and testing for pixel level classification experiments.')
print('\n')
print('ID_all: A 1-d array of size ' + str(data['Y_all'].shape) + ' containing lake IDs. This dataset was created using ' + str(np.unique(data['ID_all']).shape[0]) + ' lakes.')
print('\n')
print('QUALITY_all: A 1-d array of size ' + str(data['Y_all'].shape) + ' containing data quality information (varies between 1 - 9, 1 being highest quality and 9 being lowest quailty).')


# In[4]:


# Dataset Summary
ulabels = np.unique(data['Y_all'])
uqbits = np.unique(data['QUALITY_all'])
num_labels = np.unique(data['Y_all']).shape[0]
num_qbits = 9
label_dist = np.zeros((num_qbits,num_labels))
for i in uqbits:
    for j in ulabels:
        label_dist[i-1][j-1] = np.sum(np.logical_and(data['Y_all']==j,data['QUALITY_all']==i))


label_dist = np.transpose(label_dist)
t1 = np.array([1, 2, 3])
t2 = np.array([2, 3, 5])
p1 = plt.bar(uqbits+0.35,label_dist[0],color='blue',width=0.3)
p2 = plt.bar(uqbits+0.35,label_dist[1],color='green',bottom=label_dist[0],width=0.3)
p3 = plt.bar(uqbits+0.35,label_dist[2],color='yellow',bottom=label_dist[0]+ label_dist[1],width=0.3)
plt.ylabel('Count')
plt.xlabel('Quality Value')
plt.title('Label Distribution with respect to Quality Information')
plt.xticks(uqbits+0.5, ('1', '2', '3', '4', '5', '6', '7', '8', '9'))
plt.legend((p1[0], p2[0],p3[0]), ('Water', 'Land', 'Unknown'))
#plt.show()

print('Overall ' + str(np.sum(data['Y_all']!=3)) + ' instances have a valid label')
print('The skew (#water pixels/#land pixels) of this dataset is ' + str(np.sum(data['Y_all']==1)*1.0/np.sum(data['Y_all']==2)))


# In[10]:


# Analyzing Performance of Linear SVM


# In[5]:


# extracting data from the dictionary
X = data['X_all']
Y = data['Y_all']
ID = data['ID_all']
QUALITY = data['QUALITY_all']
water_inds = np.where(np.logical_and(Y==1, QUALITY<=9))[0]
land_inds = np.where(np.logical_and(Y==2, QUALITY<=9))[0]


# In[6]:


# selecting random samples for training and testing
num_train_samples = 1000
num_test_samples = 20000
#
cur_perm = np.random.permutation(len(water_inds))
water_train_inds  = water_inds[cur_perm[0:num_train_samples]]
water_test_inds = water_inds[cur_perm[num_train_samples:num_train_samples + num_test_samples]]
#
cur_perm = np.random.permutation(len(land_inds))
land_train_inds  = land_inds[cur_perm[0:num_train_samples]]
land_test_inds = land_inds[cur_perm[num_train_samples:num_train_samples + num_test_samples]]
#
train_X = np.append(X[water_train_inds], X[land_train_inds],axis=0)
print('Shape of Training data: ' + str(train_X.shape))
#
train_Y = np.append(np.ones((num_train_samples,1)),np.ones((num_train_samples,1))*2)
print ('Shape of Training labels: ' + str(train_Y.shape))
#
test_X = np.append(X[water_test_inds], X[land_test_inds],axis=0)
print('Shape of Test data: ' + str(test_X.shape))
#
test_Y = np.append(np.ones((num_test_samples,1)),np.ones((num_test_samples,1))*2)
print ('Shape of Test labels: ' + str(test_Y.shape))
#
#
# # In[7]:
#
#
# # Learning a Linear SVM Classifier
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(train_X, train_Y)

#
# # In[14]:
#
#
# # predicting class labels for both training and test data
pred_train_Y = clf.predict(train_X)
pred_test_Y = clf.predict(test_X)
print ('Accuracy on Training Set is: ' + str(np.sum(train_Y==pred_train_Y)*100.0/(2*num_train_samples)))
print('Accuracy on Test Set is: ' + str(np.sum(test_Y==pred_test_Y)*100.0/(2*num_test_samples)))
#


