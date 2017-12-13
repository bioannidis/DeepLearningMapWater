import os
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32"

from keras.models import *
from keras.layers import *
from keras.utils import np_utils
import tensorflow as tf
from tempfile import TemporaryFile
from scipy.io import matlab
import numpy as np
import dill
import  pickle
from scipy import stats

def create_train_test_set(train_pct,test_pct,water_inds,land_inds,X,Y):

    # selecting random samples for training and testing
    num_train_samples = int(train_pct * min(len(water_inds), len(land_inds)))
    num_test_samples = int(test_pct * min(len(water_inds), len(land_inds)))
    #
    cur_perm = np.random.permutation(len(water_inds))
    water_train_inds = water_inds[cur_perm[0:num_train_samples]]
    water_test_inds = water_inds[cur_perm[num_train_samples:num_train_samples + num_test_samples]]
    #
    cur_perm = np.random.permutation(len(land_inds))
    land_train_inds = land_inds[cur_perm[0:num_train_samples]]
    land_test_inds = land_inds[cur_perm[num_train_samples:num_train_samples + num_test_samples]]
    #
    train_X = np.append(X[water_train_inds], X[land_train_inds], axis=0)
    #
    train_Y = np.append(Y[water_train_inds], Y[land_train_inds])
    #
    test_X = np.append(X[water_test_inds], X[land_test_inds], axis=0)
    #
    test_Y = np.append(Y[water_test_inds], Y[land_test_inds], axis=0)
    #train_Y = np_utils.to_categorical(train_Y-1, 2)
    #test_Y = np_utils.to_categorical(test_Y-1, 2)
    cur_perm = np.random.permutation(len(train_Y))
    train_X = train_X[cur_perm]
    train_Y= train_Y[cur_perm]
    return train_X,train_Y,test_X,test_Y

def load_pixel_dataset():
    data = matlab.loadmat('PixelLevelDataset_Version2.mat')
    X = data['X_all']
    Y = data['Y_all']
    ID = data['ID_all']
    id=ID.astype('str')

    #append the id as the first 4 collumns in X and Y
    ID=np.core.defchararray.add(np.core.defchararray.add(np.core.defchararray.add(id[:,0],id[:,1]),id[:,2]),id[:,3])
    ID.shape += (1,)
    X=np.append(ID,X,axis=1)
    #Y = np.append(ID, Y, axis=1)
    QUALITY = data['QUALITY_all']
    return X,Y,ID,QUALITY

def clean_dataset(Y,QUALITY):
    water_inds = np.where(np.logical_and(Y == 1, QUALITY <= 9))[0]
    land_inds = np.where(np.logical_and(Y == 2, QUALITY <= 9))[0]
    return water_inds,land_inds

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def create_layer(input,input_size,output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    y = tf.nn.softmax(tf.matmul(input, W) + b)
    return y
def create_output_layer(input,input_size,output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    y = tf.matmul(input, W) + b
    return y

def create_network(input_size,output_size,v_output_layer_size):
    x = tf.placeholder(tf.float32, [None, input_size])
    input=x
    for ind in range(len(v_output_layer_size)):
        output_layer_size=v_output_layer_size[ind]
        y=create_layer(input=input,input_size=input_size,output_size=output_layer_size)
        input_size=output_layer_size
        input=y
    y=create_output_layer(input=input,input_size=input_size,output_size=output_size)
    return x,y

def create_fixed_network(input_size,output_size):
    x = tf.placeholder(tf.float32, [None, input_size])
    output_size_layer_1=40
    y1=create_layer(input=x,input_size=input_size,output_size=output_size_layer_1)
    output_size_layer_2 = 20
    y2= create_layer(input=y1, input_size=output_size_layer_1, output_size=output_size_layer_2)
    output_size_layer_3= 10
    y3 = create_layer(input=y2, input_size=output_size_layer_2, output_size=output_size_layer_3)
    y=create_output_layer(input=y3,input_size=output_size_layer_3,output_size=output_size)
    return x,y


def create_model(nClasses, input_height, input_width):
    # assert input_height%32 == 0
    # assert input_width%32 == 0

    # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, input_height, input_width), data_format='channels_first'))
    model.add(Convolution2D(32, (3, 3), activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model

def reform_labels(train_Y):
    train_Y_new = np.zeros((2, len(train_Y)))  # [[0 for col in range(2)] for row in range(len(Y))]

    # In[6]:
    for ind_array in range(len(train_Y)):
        train_Y_new[int(train_Y[ind_array] - 1)][ind_array] = 1


    train_Y_new = np.transpose(train_Y_new)

    return train_Y_new

def group_dif_lakes(X,Y,QUALITY):
    # Sort input array test w.r.t. first column that are IDs
    test_sorted = X[X[:, 0].argsort()]
    test_sortedY=Y[X[:, 0].argsort()]
    test_sortedQUALITY= QUALITY[X[:, 0].argsort()]
    # Convert the string IDs to numeric IDs
    _, numeric_ID = np.unique(test_sorted[:, 0], return_inverse=True)
    #, return_inverse=True)

    # Get the indices where shifts (IDs change) occur
    _, cut_idx = np.unique(numeric_ID, return_index=True)

    # Use the indices to split the input array into sub-arrays with common IDs
    grouped_X = np.split(test_sorted, cut_idx)[1:]
    grouped_Y = np.split(test_sortedY, cut_idx)[1:]
    grouped_quality= np.split(test_sortedQUALITY, cut_idx)[1:]
    return grouped_X,grouped_Y,grouped_quality

def create_extended_feature_set(X):
    ndvi = (X[:,1]-X[:,0])/(X[:,1]+X[:,0])
    ndvi.shape+=(1,)
    evi = 2.5*(X[:, 1] - X[:, 0]) / (X[:, 1] +6*X[:, 0]-7.5*X[:,2]+1)
    evi.shape += (1,)
    ndwi1= (X[:,1]-X[:,4])/(X[:,1]+X[:,4])
    ndwi1.shape += (1,)
    ndwi2 = (X[:, 1] - X[:, 5]) / (X[:, 1] + X[:, 5])
    ndwi2.shape += (1,)
    ndwi3 = (X[:, 1] - X[:, 6]) / (X[:, 1] + X[:, 6])
    ndwi3.shape += (1,)
    ndwi4 = (X[:, 3] - X[:, 5]) / (X[:, 3] + X[:, 5])
    ndwi4.shape += (1,)
    ndwi5 = (X[:, 0] - X[:, 5]) / (X[:, 0] + X[:, 5])
    ndwi5.shape += (1,)
    ndwi6 = (X[:, 3] - X[:, 6]) / (X[:, 3] + X[:, 6])
    ndwi6.shape += (1,)
    ndwi7 = (X[:, 3] - X[:, 1]) / (X[:, 3] + X[:, 1])
    ndwi7.shape += (1,)
    ndwi8 = (X[:, 3] - X[:, 4]) / (X[:, 3] + X[:, 4])
    ndwi8.shape += (1,)
    ndfi1 = (X[:, 0] - X[:, 6]) / (X[:, 0] + X[:, 6])
    ndfi1.shape += (1,)
    ndfi2 = (X[:, 0] - X[:, 4]) / (X[:, 0] + X[:, 4])
    ndfi2.shape += (1,)
    lswi = (X[:,1]/X[:,5])
    lswi.shape += (1,)
    extended_X = np.concatenate(
      (ndvi, evi, ndwi1, ndwi2, ndwi3, ndwi4, ndwi5, ndwi6, ndwi7, ndwi8, ndfi1, ndfi2, lswi), axis=1)
    #extended_X=np.concatenate((X,ndvi,evi,ndwi1,ndwi2,ndwi3,ndwi4,ndwi5,ndwi6,ndwi7,ndwi8,ndfi1,ndfi2,lswi),axis=1)
    #extended_X = np.concatenate(
    #   (X, ndvi), axis=1)
    return extended_X
def threshold_classifier(extended_X,threshold,feat_number):
    feat=extended_X[:,feat_number]
    water_bool=feat>threshold
    water_bool.shape += (1,)
    return water_bool
def test_thershold_classifiers():
    path = "/home/vassilis/PycharmProjects/DeepLearningMapWater/results/pxllvl_all_lakes/test_only_indices_thershold.npz"
    v_threshold=0
    X, Y, ID, QUALITY = load_pixel_dataset()
    new_X = (X[:, 1:X.shape[1]]).astype(np.float)
    new_X=create_extended_feature_set(new_X)
    pct_acc=np.zeros(new_X.shape[1]-X.shape[1]+1)
    Y=abs(Y-2)
    for ind in range(new_X.shape[1]-X.shape[1]+1):
        feat=ind+X.shape[1]-1
        water_bool=threshold_classifier(new_X, threshold=v_threshold, feat_number=feat)
        accuracy=sum((Y==(water_bool).astype(int)).astype(int))
        pct_acc[ind]=accuracy/len(Y)
    to_save=pct_acc,v_threshold
    np.savez(path, to_save)
    return pct_acc

def test_network(x,y,number_of_classes,batch_size,nbr_dataset_passes,train_X,test_X,train_Y_new,test_Y_new):

    y_ = tf.placeholder(tf.float32, [None, number_of_classes])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    tf.global_variables_initializer().run()
    # Train
    var_grad = tf.gradients(cross_entropy, [y])[0]
    tf.global_variables_initializer().run()
    for outer_iter in range(nbr_dataset_passes):
        for i in range(int(len(train_Y_new)/batch_size)):
            batch_xs = train_X[(i) * batch_size:(i + 1) * batch_size]
            batch_ys= train_Y_new[(i) * batch_size:(i + 1) * batch_size]
            var_grad_val = sess.run(var_grad, feed_dict={x: batch_xs,y_: batch_ys})
            if i % 50 == 0:
              train_accuracy = accuracy.eval(feed_dict={
                  x: batch_xs, y_: batch_ys})
              print('step %d, training accuracy %g' % (i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        pct_correct=sess.run(accuracy, feed_dict={x: test_X,
                                        y_: test_Y_new})
        print(pct_correct)
    return pct_correct



def train_test_on_all_data(train_pct,test_pct,batch_size,nbr_dataset_passes,v_output_layer_size,extended):
    X, Y, ID, QUALITY = load_pixel_dataset()

    #grouped_X, grouped_Y = group_dif_lakes(X, Y)
    water_inds, land_inds = clean_dataset(Y, QUALITY)

    new_X = (X[:, 1:X.shape[1]]).astype(np.float)
    #new_X=stats.zscore(new_X, axis=1)
    if extended:
        new_X=create_extended_feature_set(new_X)
    new_Y = Y  # (Y[:,1]).astype(np.int)
    train_X, train_Y, test_X, test_Y = create_train_test_set(train_pct, test_pct, water_inds, land_inds, new_X, new_Y)
    number_of_classes = 2
    train_Y_new = reform_labels(train_Y)
    test_Y_new = reform_labels(test_Y)

    number_of_features = (new_X).shape[1]

    x, y = create_network(number_of_features, number_of_classes, v_output_layer_size)
    pct_correct=test_network(x, y, number_of_classes, batch_size, nbr_dataset_passes, train_X, test_X, train_Y_new, test_Y_new)

    return pct_correct

def train_all_pixels_dif_conf(train_pct, test_pct, batch_size, nbr_dataset_passes,extended):
    path = "/home/vassilis/PycharmProjects/DeepLearningMapWater/results/pxllvl_all_lakes/testall6.npz"
    cells = np.array([[20,5],[10],[30],[5]])
    pct_correct=np.zeros(shape=(len(cells)))
    for ind in range(len(cells)):
        v_output_layer_size=cells[ind]
        pct_correct[ind]=train_test_on_all_data(train_pct, test_pct, batch_size, nbr_dataset_passes,v_output_layer_size,extended)

    tosave = pct_correct, train_pct, test_pct, batch_size, nbr_dataset_passes, cells,extended
    np.savez(path, tosave)
    return

def train_all_but_one_lake_test_on_the_non_trained(train_pct,test_pct,batch_size,nbr_dataset_passes,v_output_layer_size):
    path = "/home/vassilis/PycharmProjects/DeepLearningMapWater/results/pxllvl_group_lakes/testgroupall1.npz"

    X, Y, ID, QUALITY = load_pixel_dataset()

    # npzfile=np.load(path)
    # npzfile.files
    # npzfile['arr_0']
    grouped_X, grouped_Y, grouped_quality = group_dif_lakes(X, Y, QUALITY)
    pct_correct = np.zeros(shape=(len(grouped_X), 2))
    # outfile = TemporaryFile()

    for outer_ind in range(300):#len(grouped_X)):
        #test on grouped_X[outer_ind] train on the rest
        totrain_X=[]
        totrain_Y = []
        totrain_quality=[]
        grouped_help_X=copy.copy(grouped_X)
        #grouped_help_X.remove(grouped_help_X[outer_ind])
        del grouped_help_X[outer_ind]
        grouped_help_Y= copy.copy(grouped_Y)
        #grouped_help_Y.remove(grouped_help_Y[outer_ind])
        del grouped_help_Y[outer_ind]
        grouped_help_quality = copy.copy(grouped_quality)
        #grouped_help_quality.remove(grouped_help_quality[outer_ind])
        del grouped_help_quality[outer_ind]
        totrain_X=np.concatenate(grouped_help_X).reshape((X.shape[0]-grouped_X[outer_ind].shape[0],X.shape[1]))
        totrain_Y = np.concatenate(grouped_help_Y).reshape((Y.shape[0] - grouped_Y[outer_ind].shape[0], Y.shape[1]))
        totrain_quality= np.concatenate(grouped_help_quality).reshape((QUALITY.shape[0] - grouped_quality[outer_ind].shape[0], QUALITY.shape[1]))
        water_inds, land_inds = clean_dataset(totrain_Y, totrain_quality)
        new_X = (totrain_X[:, 1:X.shape[1]]).astype(np.float)
        new_Y = totrain_Y
        train_X, train_Y, test_X, test_Y = create_train_test_set(train_pct, test_pct, water_inds, land_inds, new_X,
                                                                     new_Y)
        number_of_classes = 2
        train_Y_new = reform_labels(train_Y)
        test_Y_new = reform_labels(test_Y)
        number_of_features = (new_X).shape[1]

        x, y = create_network(number_of_features, number_of_classes, v_output_layer_size)

        y_ = tf.placeholder(tf.float32, [None, number_of_classes])
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        # Train
        tf.global_variables_initializer().run()
        for outer_iter in range(nbr_dataset_passes):
            for i in range(int(len(train_Y_new) / batch_size)):
                batch_xs = train_X[(i) * batch_size:(i + 1) * batch_size]
                batch_ys = train_Y_new[(i) * batch_size:(i + 1) * batch_size]
                if i % 100 == 0 & outer_iter % 5 == 1:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch_xs, y_: batch_ys})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            pct_correct[outer_ind, 0] = sess.run(accuracy, feed_dict={x: test_X,
                                                                              y_: test_Y_new})
        print("Lake: " + str(outer_ind) + " accuracy " + str(pct_correct[outer_ind, 0]))
        pct_correct[outer_ind, 1] = sess.run(accuracy, feed_dict={
            x: (grouped_X[outer_ind][:, 1:X.shape[1]]).astype(np.float),
            y_: reform_labels(grouped_Y[outer_ind])})

    tosave = pct_correct, train_pct, test_pct, batch_size, nbr_dataset_passes, v_output_layer_size
    np.savez(path, tosave)
    return tosave

def train_one_lake_test_on_all_other(train_pct,test_pct,batch_size,nbr_dataset_passes,v_output_layer_size):

    path="/home/vassilis/PycharmProjects/DeepLearningMapWater/results/pxllvl_group_lakes/test3.npz"

    X, Y, ID, QUALITY = load_pixel_dataset()

    #npzfile=np.load(path)
    #npzfile.files
    #npzfile['arr_0']
    grouped_X, grouped_Y,grouped_quality = group_dif_lakes(X, Y,QUALITY)
    pct_correct=np.zeros(shape=(len(grouped_X),len(grouped_X)))
    nbr_pixels = np.zeros(shape=(len(grouped_X),2))
    non_valid=0
    #outfile = TemporaryFile()
    pct_excess=0.4
    min_pixels_for_train=(1/pct_excess)*batch_size

    for outer_ind in range(len(grouped_X)):
            water_inds, land_inds = clean_dataset(grouped_Y[outer_ind], grouped_quality[outer_ind])
            new_X = (grouped_X[outer_ind][:, 1:X.shape[1]]).astype(np.float)
            new_Y = grouped_Y[outer_ind]
            nbr_pixels[outer_ind, 0]=len(water_inds)
            nbr_pixels[outer_ind, 1] = len(land_inds)
            if pct_excess*min(nbr_pixels[outer_ind,:])> batch_size:
                train_X, train_Y, test_X, test_Y = create_train_test_set(train_pct, test_pct, water_inds, land_inds, new_X, new_Y)
                number_of_classes = 2
                train_Y_new = reform_labels(train_Y)
                test_Y_new = reform_labels(test_Y)
                number_of_features = (new_X).shape[1]

                x, y = create_network(number_of_features, number_of_classes,v_output_layer_size)

                y_ = tf.placeholder(tf.float32, [None, number_of_classes])
                cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
                train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                sess = tf.InteractiveSession()
                tf.global_variables_initializer().run()
                # Train
                tf.global_variables_initializer().run()
                for outer_iter in range(nbr_dataset_passes):
                    for i in range(int(len(train_Y_new) / batch_size)):
                        batch_xs = train_X[(i) * batch_size:(i + 1) * batch_size]
                        batch_ys = train_Y_new[(i) * batch_size:(i + 1) * batch_size]
                        if i % 100 == 0 & outer_iter%5 == 1:
                            train_accuracy = accuracy.eval(feed_dict={
                                x: batch_xs, y_: batch_ys})
                            print('step %d, training accuracy %g' % (i, train_accuracy))
                        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                    pct_correct[outer_ind, outer_ind]= sess.run(accuracy, feed_dict={x: test_X,
                                                                y_: test_Y_new})
                print("Lake: "+ str(outer_ind) +" accuracy " +str(pct_correct[outer_ind, outer_ind]))

                for inner_ind in range(len(grouped_X)):
                    # accuracy in other lakes
                    if outer_ind!=inner_ind:
                        pct_correct[outer_ind, inner_ind]=sess.run(accuracy, feed_dict={x: (grouped_X[inner_ind][:, 1:X.shape[1]]).astype(np.float),
                                            y_: reform_labels(grouped_Y[inner_ind])})
            else:
                pct_correct[outer_ind, outer_ind]=-1
                non_valid += 1
    tosave = nbr_pixels, pct_correct, non_valid, train_pct, test_pct, batch_size, nbr_dataset_passes, min_pixels_for_train, v_output_layer_size
    np.savez(path, tosave)
    return pct_correct,nbr_pixels,non_valid


train_pct = 0.9
test_pct = 0.1
batch_size = 5
nbr_dataset_passes = 2
v_output_layer_size = [30]
with tf.device('/device:CPU:0'):
#train_one_lake_test_on_all_other(train_pct,test_pct,batch_size,nbr_dataset_passes,v_output_layer_size)
#train_all_pixels_dif_conf(train_pct,test_pct,batch_size,nbr_dataset_passes,extended=True)
 train_all_but_one_lake_test_on_the_non_trained(train_pct,test_pct,batch_size,nbr_dataset_passes,v_output_layer_size)
#test_thershold_classifiers()