import os
import math
from matplotlib import colors
import matplotlib.pyplot as plt
from keras.models import *
import pickle as serializer
from keras.utils import np_utils
import tensorflow as tf
from tempfile import TemporaryFile
from scipy.io import matlab
import numpy as np
from keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Dropout, \
    BatchNormalization
from keras.models import Sequential
from sklearn import preprocessing

def load_img_dataset(dataset_name):
    image_names = os.listdir(dataset_name)
    ID_name = []
    for image_file in image_names:
        ID_name.append(image_file.replace('data_', '').replace('.mat', ''))
    return ID_name

def replace_zero_with_ones(z):
    zero_ind=np.where(z==0)
    for ind in range((zero_ind[0].shape[0])):
       z[zero_ind[0][ind],zero_ind[1][ind],zero_ind[2][ind]]=1
    return z

def create_extended_feature_set(X):
    correction=0
    ndvi = (X[:,:,:,1]-X[:,:,:,0])/replace_zero_with_ones(X[:,:,:,1]+X[:,:,:,0]+correction)
    ndvi.shape+=(1,)
    evi = 2.5*(X[:,:,:, 1] - X[:,:,:, 0]) / replace_zero_with_ones(X[:,:,:, 1] +6*X[:,:,:, 0]-7.5*X[:,:,:,2]+1)
    evi.shape += (1,)
    ndwi1= (X[:,:,:,1]-X[:,:,:,4])/replace_zero_with_ones(X[:,:,:,1]+X[:,:,:,4]+correction)
    ndwi1.shape += (1,)
    ndwi2 = (X[:,:,:, 1] - X[:,:,:, 5]) / replace_zero_with_ones(X[:,:,:, 1] + X[:,:,:, 5]+correction)
    ndwi2.shape += (1,)
    ndwi3 = (X[:,:,:, 1] - X[:,:,:, 6]) / replace_zero_with_ones(X[:,:,:, 1] + X[:,:,:, 6]+correction)
    ndwi3.shape += (1,)
    ndwi4 = (X[:,:,:, 3] - X[:,:,:, 5]) / replace_zero_with_ones(X[:,:,:, 3] + X[:,:,:, 5]+correction)
    ndwi4.shape += (1,)
    ndwi5 = (X[:,:,:, 0] - X[:,:,:, 5]) / replace_zero_with_ones(X[:,:,:, 0] + X[:,:,:, 5]+correction)
    ndwi5.shape += (1,)
    ndwi6 = (X[:,:,:, 3] - X[:,:,:, 6]) / replace_zero_with_ones(X[:,:,:, 3] + X[:,:,:, 6]+correction)
    ndwi6.shape += (1,)
    ndwi7 = (X[:,:,:, 3] - X[:,:,:, 1]) / replace_zero_with_ones(X[:,:,:, 3] + X[:,:,:, 1]+correction)
    ndwi7.shape += (1,)
    ndwi8 = (X[:,:,:, 3] - X[:,:,:, 4]) / replace_zero_with_ones(X[:,:,:, 3] + X[:,:,:, 4]+correction)
    ndwi8.shape += (1,)
    ndfi1 = (X[:,:,:, 0] - X[:,:,:, 6]) / replace_zero_with_ones(X[:,:,:, 0] + X[:,:,:, 6]+correction)
    ndfi1.shape += (1,)
    ndfi2 = (X[:,:,:, 0] - X[:,:,:, 4]) / replace_zero_with_ones(X[:,:,:, 0] + X[:,:,:, 4]+correction)
    ndfi2.shape += (1,)
    lswi = (X[:,:,:,1]/replace_zero_with_ones(X[:,:,:,5]))
    lswi.shape += (1,)
    #extended_X = np.concatenate(
    # (ndvi, evi, ndwi1, ndwi2, ndwi3, ndwi4, ndwi5, ndwi6, ndwi7, ndwi8, ndfi1, ndfi2, lswi), axis=1)
    #X=np.squeeze(X)
    extended_X=np.concatenate((X,ndvi,evi,ndwi1,ndwi2,ndwi3,ndwi4,ndwi5,ndwi6,ndwi7,ndwi8,ndfi1,ndfi2,lswi),axis=3)
    #extended_X = np.concatenate(
    #   (X, ndvi), axis=1)
    #X = extended_X.reshape(1, extended_X.shape[0], extended_X.shape[1], extended_X.shape[2])
    return extended_X


def patch_creator(image, X_array, Y_array, horizontal, vertical):
    # Get Features (pixels) and labels for the image
    pixels = image['X']
    labels = image['Y']

    # Used to encode
    lb = preprocessing.LabelBinarizer()
    lb.fit([1, 2])

    # check to see if patch size is bigger than image
    if labels.shape > (horizontal, vertical):

        # Find how many patches can be achieved vertically and horizontally
        Num_Horizontal = math.floor(labels.shape[1] / horizontal)
        Num_Vert = math.floor(labels.shape[0] / vertical)

        # initalize vertical position
        vert_pos1 = 0
        vert_pos2 = vertical - 1

        # initalize horizontal position
        horz_pos1 = horizontal * -1
        horz_pos2 = -1

        for vertical_shift in range(Num_Vert):

            # \Get all horizontal shifts for each vertical shift first
            for horizontal_shift in range(Num_Horizontal):

                horz_pos1 = horz_pos1 + horizontal
                horz_pos2 = horz_pos2 + horizontal

                # slices image on dimensions to get patch
                X_patch = pixels[vert_pos1:vert_pos2, horz_pos1:horz_pos2, :].reshape(1, vertical - 1, horizontal - 1,  7)

                #if extended:
                #    X_patch=create_extended_feature_set(X_patch)
                # Binzaration and Encode classes (3)
                Y_patch = labels[vert_pos1:vert_pos2, horz_pos1:horz_pos2].reshape((vertical - 1) * (horizontal - 1))
                #encoded_Y_patch = lb.transform(Y_patch)
                encoded_Y_patch = reform_labels(Y_patch)
                encoded_Y_patch = encoded_Y_patch.reshape(1, encoded_Y_patch.shape[0], encoded_Y_patch.shape[1])
                # If you have an empty array, create a new one,
                # Else keep adding patches and info

                water_pxls,land_pxls=sum(sum(encoded_Y_patch))
                # discard noninformative patches that dont contain water
                if True:#water_pxls!=0:
                    if X_array == []:
                        X_array = np.array(X_patch)
                    elif X_array != []:
                        X_array = np.vstack((X_array, X_patch))

                    if Y_array == []:
                        Y_array = np.array(encoded_Y_patch)
                    elif Y_array != []:
                        Y_array = np.vstack((Y_array, encoded_Y_patch))

            # Shift vertical Patch
            vert_pos1 = vert_pos1 + vertical
            vert_pos2 = vert_pos2 + vertical

            # reset horizontal position
            horz_pos1 = (horizontal * -1)
            horz_pos2 = -1

    return (X_array, Y_array)

def select_best_threshold(thresholds,extended_X,feat_number,test_Y):
    test_accuracy=np.zeros(len(thresholds))
    water_recall=np.zeros(len(thresholds))
    water_precision=np.zeros(len(thresholds))
    earth_precision=np.zeros(len(thresholds))
    earth_recall=np.zeros(len(thresholds))
    for thresh_ind in range(len(thresholds)):
        pred_int_Y=threshold_classifier(extended_X,thresholds[thresh_ind],feat_number)
        test_accuracy[thresh_ind], water_recall[thresh_ind], water_precision[thresh_ind], earth_precision[thresh_ind], \
        earth_recall[thresh_ind] = perf_measure(test_Y,pred_int_Y)

    return

def reform_labels(train_Y):
    train_Y_new = np.zeros((2, len(train_Y)))  # [[0 for col in range(2)] for row in range(len(Y))]

    # In[6]:
    for ind_array in range(len(train_Y)):
        train_Y_new[int(train_Y[ind_array])-1][ind_array] = 1


    train_Y_new = np.transpose(train_Y_new)

    return train_Y_new

def reform_labels_back(train_Y):
    train_Y_new = np.zeros((1, (train_Y).shape[0]))  # [[0 for col in range(2)] for row in range(len(Y))]

    return train_Y_new

# Function to create model
def normalize_array(x):
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)
    #x = (x ) / (x_max)
    x = (x - x_min) / (x_max - x_min)
    return x

def model_build(train_X,deep):
    if deep:
        model=deep_model_build(train_X)
    else:
        model=simple_model_build(train_X)

    return model

def simple_model_build(train_X):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=train_X.shape[1:]))
    model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=.001))
    #model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2DTranspose(2, (4, 4), activation='relu'))
    model.add(Reshape((train_X.shape[1]*train_X.shape[2], 2)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],sample_weight_mode="temporal")

    model.summary()
    return (model)


# Function to create model
def deep_model_build(train_X):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=train_X.shape[1:]))
    model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=.001))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(112, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(112, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2DTranspose(112, (4, 4), activation='relu'))
    model.add(Conv2DTranspose(64, (3, 3), activation='relu'))
    model.add(Conv2DTranspose(2, (2, 2), activation='relu'))
    model.add(Reshape(((train_X.shape[1] ** 2), 2)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],sample_weight_mode="temporal")
    model.summary()
    return (model)
def create_dataset(ID_name,train_pct,val_pct,patch_size):
    #train_X = train_Y = test_Y = test_X = valid_X = valid_Y =\
    total_X=total_Y= []
    for  ind in ID_name:
        data = matlab.loadmat(dataset_name + '/data_' + ind + '.mat')
        total_X, total_Y = patch_creator(data, total_X, total_Y, patch_size, patch_size)
    water_pxl,land_pxl=sum(sum(total_Y))
    land_pct=land_pxl/(land_pxl+water_pxl)
    water_pct = water_pxl / (land_pxl + water_pxl)
    permutation = np.random.permutation(total_X.shape[0])
    total_X=total_X[permutation,:,:,:]
    total_Y=total_Y[permutation,:,:]
    length=total_X.shape[0]
    train_X=total_X[0:int(train_pct * length),:,:,:]
    train_Y=total_Y[0:int(train_pct * length),:,:]
    valid_X = total_X[int(train_pct * length) + 1:int(train_pct * length + val_pct * length), :, :, :]
    valid_Y = total_Y[int(train_pct * length) + 1:int(train_pct * length + val_pct * length), :, :]
    test_X = total_X[int(train_pct * length) + int(val_pct * length) + 1:length, :, :, :]
    test_Y = total_Y[int(train_pct * length) + int(val_pct * length) + 1:length, :, :]


    return train_X,test_X,valid_X,train_Y,test_Y,valid_Y

def get_var_value(filename="varstore.dat"):
    with open(filename, "r+") as f:
        val = int(f.read()) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

def perf_measure(y_actual, y_hat):
    TE=TW=FE=FW=0
    for ind_i in range(y_hat.shape[0]):
        for ind_j in range(y_hat.shape[1]):
            if y_actual[ind_i,ind_j,1]==y_hat[ind_i,ind_j,1]==1:
                TE+=1
            elif y_actual[ind_i, ind_j, 0] == y_hat[ind_i, ind_j, 0] == 1 :
                TW += 1
            elif y_actual[ind_i, ind_j, 1] != y_hat[ind_i, ind_j, 1] & y_hat[ind_i, ind_j, 1]  == 1:
                FE += 1
            elif y_actual[ind_i, ind_j, 0] != y_hat[ind_i, ind_j, 0] & y_hat[ind_i, ind_j, 0]  == 1:
                FW += 1
    water_pxl, land_pxl = sum(sum(y_actual))
    disagrement = (y_hat != (y_actual.astype(bool))).astype(int)
    pct_error = sum(sum(disagrement))[1] / (y_hat.shape[0] * y_hat.shape[1])
    test_accuracy = 1 - pct_error
    water_recall = TW / water_pxl
    if TW + FW == 0:
        FW = 1
    water_precision = TW / (TW + FW)
    earth_recall = TE / land_pxl
    earth_precision = TE / (TE + FE)


    return test_accuracy,water_recall,water_precision,earth_precision,earth_recall

def create_weight_matrix(weight,Y):
    weight_matrix=np.zeros((Y.shape[0],Y.shape[1]))
    for ind_i in range(Y.shape[0]):
        for ind_j in range(Y.shape[1]):
            if Y[ind_i,ind_j,0]==1:
                weight_matrix[ind_i,ind_j]=weight
            else:
                weight_matrix[ind_i, ind_j] = 1
    return weight_matrix

def plot_gray_scale(y,patch_size,fig_index,path):
    g=np.zeros((patch_size,patch_size))
    cmap = colors.ListedColormap(['green', 'blue'])
    for ind_i in range(patch_size):
        for ind_j in range(patch_size):
            if y[ind_j+ind_i*(patch_size),0]==1:
                g[ind_i, ind_j] =1
            else:
                g[ind_i, ind_j] =0
    plt.figure(fig_index)
    plt.imshow(g, cmap=cmap, interpolation='nearest')
    if path!=[]:
        plt.savefig(path)

    #plt.show()
    return

def save_to_file(file_name,extended,normalize,batch_size,patch_size,epochs,train_pct,val_pct_,water_recall,earth_recall,
                 earth_precision,water_precision,test_accuracy,weigth,len,deep):
    with open(file_name,'w') as f:
        f.write("extended, "+str(extended)+"\n")
        f.write("normalize, "+str(normalize)+"\n")
        f.write("batch_size, "+str(batch_size)+"\n")
        f.write("patch_size, "+str(patch_size)+"\n")
        f.write("weight, "+str(weigth)+"\n")
        f.write("epochs, "+str(epochs)+"\n")
        f.write("train_pct, "+str(train_pct)+"\n")
        f.write("val_pct, "+str(val_pct_)+"\n")
        f.write("water_recall, "+str(water_recall)+"\n")
        f.write("earth_recall, "+str(earth_recall)+"\n")
        f.write("earth_precision, "+str(earth_precision)+"\n")
        f.write("water_precision, "+str(water_precision)+"\n")
        f.write("test_accuracy, "+str(test_accuracy)+"\n")
        f.write("total examples, " + str(len) + "\n")
        f.write("Deep model is used, " + str(deep) + "\n")
    return

def threshold_classifier(extended_X,threshold,feat_number):
    feat=extended_X[:,:,:,feat_number]
    water_bool=feat>threshold
    #water_bool.shape += (1,)
    water_int=water_bool.astype(int)
    pred_labels=water_int.reshape(water_int.shape[0],water_int.shape[1]*water_int.shape[2])
    pred_labels=abs(pred_labels-2)
    pred_bool_Y=np.zeros((pred_labels.shape[0],pred_labels.shape[1],2))
    for ind in range(pred_labels.shape[0]):
        pred_bool_Y[ind,:,:]=reform_labels(pred_labels[ind,:])
    pred_int_Y= pred_bool_Y.astype(int)
    return pred_int_Y
def create_plot_patch(ind):
    data = matlab.loadmat(dataset_name + '/data_' + ind + '.mat')
    to_plot_X, to_plot_Y = patch_creator(data, [],[], patch_size, patch_size)
    return to_plot_X, to_plot_Y
def reform_pred_to_meas(predict_Y):
    bool_predict_Y = np.zeros_like(predict_Y)

    for ind in range(predict_Y.shape[0]):
        bool_predict_Y[ind, :, :] = (predict_Y[ind, :, :] == predict_Y[ind, :, :].max(axis=1)[:, None])
    int_predict_Y = bool_predict_Y.astype(int)
    return int_predict_Y

def train_and_test_model(file_name,extended,normalize,batch_size,patch_size,epochs,train_pct,val_pct,
                         weigth,dataset_name,deep,plot,image_path):
    ID_name = load_img_dataset(dataset_name)
    if plot:
        img_to_plot=ID_name[140]
        ID_name.remove(img_to_plot)
        to_plot_X,to_plot_Y=create_plot_patch(img_to_plot)

    train_X, test_X, valid_X, train_Y, test_Y, valid_Y = create_dataset(ID_name, train_pct, val_pct, patch_size)
    len = train_X.shape[0] + valid_X.shape[0] + test_X.shape[0]

    # Initalize and create Patches
    thresholds=[0,0.2,0.4,0.6,0.8]
    if extended:
        train_X = create_extended_feature_set(train_X)
        valid_X = create_extended_feature_set(valid_X)
        test_X = create_extended_feature_set(test_X)
        pred_int_Y = select_best_threshold(thresholds,train_X,10,train_Y)
        if plot:
            to_plot_X = create_extended_feature_set(to_plot_X)

    if normalize:
        test_X = normalize_array(test_X)
        valid_X = normalize_array(valid_X)
        train_X = normalize_array(train_X)
    # Build Model

    model = model_build(train_X,deep)
    # weight for class imbalance
    weigth_matrix = create_weight_matrix(weight=weigth, Y=train_Y)
    model.fit(train_X, train_Y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(valid_X, valid_Y), sample_weight=weigth_matrix)
    predict_Y = model.predict(test_X)
    int_predict_Y=reform_pred_to_meas(predict_Y)
    #test_accuracyB, water_recallB, water_precisionB, earth_precisionB, earth_recallB = perf_measure(test_Y, pred_int_Y)
    test_accuracy, water_recall, water_precision, earth_precision, earth_recall= perf_measure(test_Y, int_predict_Y)


    save_to_file(file_name, extended, normalize, batch_size, patch_size, epochs, train_pct, val_pct, water_recall,
                 earth_recall,
                 earth_precision, water_precision, test_accuracy, weigth, len,deep)
    if plot:
        predict_to_plot_Y=model.predict(to_plot_X)
        int_predict_to_plot_Y=reform_pred_to_meas(predict_to_plot_Y)
        plot_gray_scale(to_plot_Y[5, :, :], patch_size - 1, 1,[])
        plot_gray_scale(int_predict_to_plot_Y[5, :, :], patch_size - 1, 2,image_path)



    return


dataset_name='ImageLevelDataset_Version2'
patch_size=33
batch_size=3
epochs=20
extended=False
plot=True

your_counter = get_var_value()
deep=False
image_path="/home/vassilis/PycharmProjects/DeepLearningMapWater/results/imglvl/test"+str(your_counter)+".png"
file_name="/home/vassilis/PycharmProjects/DeepLearningMapWater/results/imglvl/test"+str(your_counter)+".txt"
normalize=True

train_pct=0.7
val_pct=0.05
weigth=5

train_and_test_model(file_name,extended,normalize,batch_size,patch_size,epochs,train_pct,val_pct,
                         weigth,dataset_name,deep,plot,image_path)




