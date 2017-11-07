# # Notes on Pixel Level Dataset Version 1

# importing modules
from scipy.io import matlab
import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
#get_ipython().magic('matplotlib inline')
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#loading data from the server. This will download the dataset in your curernt directory and then load it.
url = 'http://umnlcc.cs.umn.edu/WaterDatasets/PixelLevelDataset_Version1.mat'
urllib.request.urlretrieve(url,'PixelLevelDataset_Version1.mat')
data = matlab.loadmat('PixelLevelDataset_Version1.mat')

#Remove unecessary columns in dataset
for key in ['__header__', '__version__', '__globals__']:
    if key in data:
        del data[key]

#Seperate out columns in X, Y, and ID
X = pd.DataFrame.from_dict(data['X_all'])
X.columns = ['X1','X2','X3','X4','X5','X6','X7']
ID = pd.DataFrame.from_dict(data['ID_all'])
ID.columns =['ID']
Y = pd.DataFrame.from_dict(data['Y_all'])
Y.columns = ['Label']

#Join the X & Y dataframes together
dataset = ID.join(X)
dataset= dataset.join(Y)

#Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state = 0)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, np.ravel(y_train))
y_pred = classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print ( classifier.score(X_test, y_test))
