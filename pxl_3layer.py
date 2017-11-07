

# importing modules and data (execute if using python 3.x)
# importing modules
from scipy.io import matlab
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tensorflow as tf
import tempfile

from sklearn import svm
# get_ipython().magic(u'matplotlib inline')







def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 7, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([7, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
#  with tf.name_scope('pool1'):
#    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([7, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1 , W_conv2) + b_conv2)

  # Second pooling layer.
#  with tf.name_scope('pool2'):
#    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_conv2, [-1, 7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)














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



# selecting random samples for training and testing
num_train_samples = 1000
num_test_samples = 10000
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


# Create the model
x = tf.placeholder(tf.float32, [None, 7])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch_xs = train_X[(i) * 10:(i + 1) * 10]
        batch_ys = train_Y_new[(i) * 10:(i + 1) * 10]
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: test_X, y_: test_Y_new, keep_prob: 1.0}))


