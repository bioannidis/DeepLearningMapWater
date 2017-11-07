

# importing modules and data (execute if using python 3.x)
# importing modules
from scipy.io import matlab
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tensorflow as tf
import collections
from sklearn import svm
# get_ipython().magic(u'matplotlib inline')
FLAGS = None
def train():
  # Import data
  num_train_samples=FLAGS.num_train_samples
  num_test_samples=FLAGS.num_test_samples
  data=load_data()
  processed_data=process_data(data, num_train_samples, num_test_samples)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()
def load_data():
#loading data from the server. This will download the dataset in your curernt directory and then load it.
    url = 'http://umnlcc.cs.umn.edu/WaterDatasets/PixelLevelDataset_Version1.mat'
    urllib.request.urlretrieve(url,'PixelLevelDataset_Version1.mat')
    data = matlab.loadmat('PixelLevelDataset_Version1.mat')
    return data

def process_data(data,num_train_samples,num_test_samples):
    X = data['X_all']
    Y = data['Y_all']
    ID = data['ID_all']
    QUALITY = data['QUALITY_all']
    water_inds = np.where(np.logical_and(Y==1, QUALITY<=9))[0]
    land_inds = np.where(np.logical_and(Y==2, QUALITY<=9))[0]
    #num_train_samples = 200000
    #num_test_samples = 100000
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

    test_Y = np.append(np.ones((num_test_samples,1)),np.ones((num_test_samples,1))*2)
    print ('Shape of Test labels: ' + str(test_Y.shape))

    train_Y_new=np.zeros((2,len(train_Y))) #[[0 for col in range(2)] for row in range(len(Y))]

    for ind_array in range(len(train_Y)):
        train_Y_new[int(train_Y[ind_array]-1)][ind_array]=1


    test_Y_new=np.zeros((2,len(test_Y))) #[[0 for col in range(2)] for row in range(len(Y))]
    for ind_array in range(len(test_Y)):
        test_Y_new[int(test_Y[ind_array]-1)][ind_array]=1

    test_Y_new=np.transpose(test_Y_new)
    train_Y_new=np.transpose(train_Y_new)
    cur_perm = np.random.permutation(len(train_X))
    train_X=train_X[cur_perm]
    train_Y_new=train_Y_new[cur_perm]
    Processed_data = collections.namedtuple('Processed_data', ['tr_x','tr_y','test_x','test_y'])
    processed_data=Processed_data(train_X,train_Y_new,test_X,test_Y_new)
    return processed_data

#
#
# # In[7]:
#
#
# Create the model
x = tf.placeholder(tf.float32, [None, 7])
h1 = tf.nn.softmax(nn_layer(x, 7, 30, 'layer1'))
y=nn_layer(h1,30,2,'layer2', act=tf.identity)


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

with tf.name_scope('cross_entropy'):
  # The raw formulation of cross-entropy,
  #
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                               reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the
  # raw outputs of the nn_layer above, and then average across
  # the batch.
  diff = tf.nn.softmax_cross_entropy_with_logits(targets=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)


#train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
batch_size=1000
for index_train in range(int(num_train_samples/batch_size)):
    batch_xs= train_X[(index_train)*batch_size:(index_train+1)*batch_size]
    batch_ys =  train_Y_new[(index_train)*batch_size:(index_train+1)*batch_size]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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


