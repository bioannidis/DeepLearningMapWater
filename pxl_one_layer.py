

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
def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#loading data from the server. This will download the dataset in your curernt directory and then load it.
#url = 'http://umnlcc.cs.umn.edu/WaterDatasets/PixelLevelDataset_Version2.mat'
#urllib.request.urlretrieve(url,'PixelLevelDataset_Version2.mat')

data = matlab.loadmat('PixelLevelDataset_Version2.mat')
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
#label_dist = np.zeros((num_qbits,num_labels))
#for i in uqbits:
#    for j in ulabels:
#        label_dist[i-1][j-1] = np.sum(np.logical_and(data['Y_all']==j,data['QUALITY_all']==i))


#label_dist = np.transpose(label_dist)
#t1 = np.array([1, 2, 3])
#t2 = np.array([2, 3, 5])
#p1 = plt.bar(uqbits+0.35,label_dist[0],color='blue',width=0.3)
#p2 = plt.bar(uqbits+0.35,label_dist[1],color='green',bottom=label_dist[0],width=0.3)
#p3 = plt.bar(uqbits+0.35,label_dist[2],color='yellow',bottom=label_dist[0]+ label_dist[1],width=0.3)
#plt.ylabel('Count')
#plt.xlabel('Quality Value')
#plt.title('Label Distribution with respect to Quality Information')
#plt.xticks(uqbits+0.5, ('1', '2', '3', '4', '5', '6', '7', '8', '9'))
#plt.legend((p1[0], p2[0],p3[0]), ('Water', 'Land', 'Unknown'))
#plt.show()

#print('Overall ' + str(np.sum(data['Y_all']!=3)) + ' instances have a valid label')
#print('The skew (#water pixels/#land pixels) of this dataset is ' + str(np.sum(data['Y_all']==1)*1.0/np.sum(data['Y_all']==2)))


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


train_pct=0.8
test_pct=0.2
# selecting random samples for training and testing
num_train_samples = int(train_pct*min(len(water_inds),len(land_inds)))
num_test_samples = int(test_pct*min(len(water_inds),len(land_inds)))
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


train_Y_new=np.zeros((2,len(train_Y))) #[[0 for col in range(2)] for row in range(len(Y))]

# In[6]:
for ind_array in range(len(train_Y)):
    train_Y_new[int(train_Y[ind_array]-1)][ind_array]=1


test_Y_new=np.zeros((2,len(test_Y))) #[[0 for col in range(2)] for row in range(len(Y))]

# In[6]:
for ind_array in range(len(test_Y)):
    test_Y_new[int(test_Y[ind_array]-1)][ind_array]=1

test_Y_new=np.transpose(test_Y_new)
train_Y_new=np.transpose(train_Y_new)
cur_perm = np.random.permutation(len(train_X))
train_X=train_X[cur_perm]
train_Y_new=train_Y_new[cur_perm]
#
#
# # In[7]:
#
#
# Create the model
x = tf.placeholder(tf.float32, [None, 7])
W = weight_variable([7, 30])
b = bias_variable([30])
y1 = tf.nn.softmax(tf.matmul(x, W) + b)

W1= weight_variable([30, 10])
b1= bias_variable([10])
y2 = tf.nn.softmax(tf.matmul(y1, W1) + b1)
#y = (tf.matmul(y1, W1) + b1)

W2= weight_variable([10,2])
b2= bias_variable([2])
y=tf.matmul(y2, W2) + b2
#W1= tf.Variable(tf.zeros([7, 2]))
#b1= tf.Variable(tf.zeros([2]))

#y = tf.nn.softmax(tf.matmul(x, W1) + b1)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
batch_size=10
tf.global_variables_initializer().run()
for i in range(int(num_train_samples/batch_size)):
    batch_xs = train_X[(i) * batch_size:(i + 1) * batch_size]
    batch_ys= train_Y_new[(i) * batch_size:(i + 1) * batch_size]
    if i % 50 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch_xs, y_: batch_ys})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print(sess.run(accuracy, feed_dict={x: test_X,
                                    y_: test_Y_new}))

#
# # In[14]:
#
#
# # predicting class labels for both training and test data
#pred_train_Y = clf.predict(train_X)
#pred_test_Y = clf.predict(test_X)
#print ('Accuracy on Training Set is: ' + str(np.sum(train_Y==pred_train_Y)*100.0/(2*num_train_samples)))
#print('Accuracy on Test Set is: ' + str(np.sum(test_Y==pred_test_Y)*100.0/(2*num_test_samples)))
#


