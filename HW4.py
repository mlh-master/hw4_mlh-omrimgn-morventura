#!/usr/bin/env python
# coding: utf-8

# # HW: X-ray images classification
# --------------------------------------

# Before you begin, open Mobaxterm and connect to triton with the user and password you were give with. Activate the environment `2ndPaper` and then type the command `pip install scikit-image`.

# In this assignment you will be dealing with classification of 32X32 X-ray images of the chest. The image can be classified into one of four options: lungs (l), clavicles (c), and heart (h) and background (b). Even though those labels are dependent, we will treat this task as multiclass and not as multilabel. The dataset for this assignment is located on a shared folder on triton (`/MLdata/MLcourse/X_ray/'`).

# In[ ]:


import os
import numpy as np
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *

from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread

from skimage.transform import rescale, resize, downscale_local_mean
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"


import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[ ]:


def preprocess(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn),1)
        img = resize(src,(32,32),order = 3)
        
        images[ii,:,:,0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


# In[ ]:


def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        images[ii,:,:,0] = imread(os.path.join(datapath, fn),1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    return images, Y


# In[ ]:


#Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train, BaseY_train = preprocess_train_and_val(train_path)
BaseX_val, BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)


# In[ ]:


keras.backend.clear_session()


# ### PART 1: Fully connected layers 
# --------------------------------------
# 
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification.
# Use ReLU activation functions for the hidden layers and He_normal for initialization.
# Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*

#--------------------------Impelment your code here:-------------------------------------
dropout = 0.2  # as in tutorial
model_relu = Sequential(name='model_relu')
model_relu.add(Flatten(input_shape=[32, 32, 1]))
model_relu.add(Dense(300, kernel_initializer='he_normal', activation='relu'))
model_relu.add(Dropout(dropout))
model_relu.add(Dense(150, kernel_initializer="he_normal", activation='relu'))
model_relu.add(Dropout(dropout))
model_relu.add(Dense(4, activation='softmax'))
#----------------------------------------------------------------------------------------


model_relu.summary()

#Inputs: 
input_shape = (32, 32, 1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task.
# Train your model on the training set and evaluate the model on the testing set.
# Print the accuracy and loss over the testing set.

#--------------------------Impelment your code here:-------------------------------------
# compiling
model_relu.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

# saving initial weights
if not ("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "init_weights_relu1.h5"
model_path = os.path.join(save_dir, model_name)
model_relu.save(model_path)
print('Saved initialized model at %s ' % model_path)

# training the model
history = model_relu.fit(BaseX_train, BaseY_train,
                         batch_size=batch_size, epochs=epochs,
                         verbose=2,
                         validation_data=(X_test, Y_test))

print(history.history.keys())
# accuracy and loss (based on tutorial 11)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history.history['acc'])
axs[0].plot(history.history['val_acc'])
axs[0].set_title('relu model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='lower right')


axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('relu model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

# save the model (final weights)
model_name = "final_weights_relu1.h5"
model_path = os.path.join(save_dir, model_name)
model_relu.save(model_path)
print('Saved trained model at %s ' % model_path)

# evaluating
loss_and_metrics = model_relu.evaluate(X_test, Y_test, verbose=2)
print("Test Loss is {:.2f} ".format(loss_and_metrics[0]))
print("Test Accuracy is {:.2f} %".format(100*loss_and_metrics[1]))

#----------------------------------------------------------------------------------------

# Task 2:*Activation functions.*
# Change the activation functions to LeakyRelu or tanh or sigmoid.
# Name the new model `new_a_model`.
# Explain how it can affect the model.*

#--------------------------Impelment your code here:-------------------------------------
new_activation = 'tanh'

dropout = 0.2
new_a_model = Sequential(name='new_a_model')
new_a_model.add(Flatten(input_shape=[32, 32, 1]))
new_a_model.add(Dense(300, kernel_initializer='he_normal', activation=new_activation))
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(150, kernel_initializer="he_normal", activation=new_activation))
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(4, activation='softmax'))
#----------------------------------------------------------------------------------------

new_a_model.summary()


# ---
# ***Task 3:***</span> *Number of epochs.*
# 
# Train the new model using 25 and 40 epochs.
# What difference does it makes in term of performance?
# Remember to save the compiled model for having initialized weights for every run as we did in tutorial 12.
# Evaluate each trained model on the test set*
# 
# ---

#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#--------------------------Impelment your code here:-------------------------------------
# compiling
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

# saving initial weights
if not ("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "init_weights_act_25.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved initialized model at %s ' % model_path)

# training the model
history_1 = new_a_model.fit(BaseX_train, BaseY_train,
                         batch_size=batch_size, epochs=epochs,
                         verbose=2,
                         validation_data=(X_test, Y_test))

print(history_1.history.keys())
# accuracy and loss (based on tutorial 11)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history_1.history['acc'])
axs[0].plot(history_1.history['val_acc'])
axs[0].set_title('new_a model accuracy - 25 epochs')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='lower right')


axs[1].plot(history_1.history['loss'])
axs[1].plot(history_1.history['val_loss'])
axs[1].set_title('new_a model loss - 25 epochs')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

# save the model (final weights)
model_name = "final_weights_act_25.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved trained model at %s ' % model_path)

# evaluating
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss is {:.2f} ".format(loss_and_metrics[0]))
print("Test Accuracy is {:.2f} %".format(100*loss_and_metrics[1]))
#-----------------------------------------------------------------------------------------

#Inputs: 
input_shape = (32, 32, 1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

#--------------------------Impelment your code here:-------------------------------------
# compiling
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

# saving initial weights
if not ("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "init_weights_act_40.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved initialized model at %s ' % model_path)

# training the model
history_2 = new_a_model.fit(BaseX_train, BaseY_train,
                         batch_size=batch_size, epochs=epochs,
                         verbose=2,
                         validation_data=(X_test, Y_test))

print(history_2.history.keys())
# accuracy and loss (based on tutorial 11)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history_2.history['acc'])
axs[0].plot(history_2.history['val_acc'])
axs[0].set_title('new_a model accuracy - 40 epochs')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='lower right')


axs[1].plot(history_2.history['loss'])
axs[1].plot(history_2.history['val_loss'])
axs[1].set_title('new_a model loss - 40 epochs')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

# save the model (final weights)
model_name = "final_weights_act_40.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved trained model at %s ' % model_path)

# evaluating
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss is {:.2f} ".format(loss_and_metrics[0]))
print("Test Accuracy is {:.2f} %".format(100*loss_and_metrics[1]))
#-----------------------------------------------------------------------------------------


# ---
# >***Task 4:*Mini-batches.*
# 
# Build the `model_relu` again and run it with a batch size of 32 instead of 64.
# What are the advantages of the mini-batch vs. SGD?*
# 
# ---

keras.backend.clear_session()


#--------------------------Impelment your code here:-------------------------------------
dropout = 0.2  # as in tutorial
model_relu = Sequential(name='model_relu')
model_relu.add(Flatten(input_shape=[32, 32, 1]))
model_relu.add(Dense(300, kernel_initializer='he_normal', activation='relu'))
model_relu.add(Dropout(dropout))
model_relu.add(Dense(150, kernel_initializer="he_normal", activation='relu'))
model_relu.add(Dropout(dropout))
model_relu.add(Dense(4, activation='softmax'))
#----------------------------------------------------------------------------------------

batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#--------------------------Impelment your code here:-------------------------------------
# compiling
model_relu.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

# saving initial weights
if not ("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "init_weights_relu_32.h5"
model_path = os.path.join(save_dir, model_name)
model_relu.save(model_path)
print('Saved initialized model at %s ' % model_path)

# training the model
history_3 = model_relu.fit(BaseX_train, BaseY_train,
                         batch_size=batch_size, epochs=epochs,
                         verbose=2,
                         validation_data=(X_test, Y_test))

print(history_3.history.keys())
# accuracy and loss (based on tutorial 11)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history_3.history['acc'])
axs[0].plot(history_3.history['val_acc'])
axs[0].set_title('relu 32 model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='lower right')


axs[1].plot(history_3.history['loss'])
axs[1].plot(history_3.history['val_loss'])
axs[1].set_title('relu 32 model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

# save the model (final weights)
model_name = "final_weights_relu_32.h5"
model_path = os.path.join(save_dir, model_name)
model_relu.save(model_path)
print('Saved trained model at %s ' % model_path)

# evaluating
loss_and_metrics = model_relu.evaluate(X_test, Y_test, verbose=2)
print("Test Loss is {:.2f} ".format(loss_and_metrics[0]))
print("Test Accuracy is {:.2f} %".format(100*loss_and_metrics[1]))

print(' ReLU 32, epoch 25:\n', 'train accuracy: ',
      history_3.history['acc'][24], '\n',
      'test accuracy: ', history_3.history['val_acc'][24], '\n',
      'ReLU 64, epoch 25:\n', 'train accuracy: ',
      history.history['acc'][24], '\n',
      'test accuracy: ', history.history['val_acc'][24], '\n'
      )

# print(' ReLU 32, 25 epoches:\n', 'Test Loss: ',
#       np.mean(history_3.history['val_loss'][:24]), '\n',
#       'Test Accuracy: ', np.mean(history_3.history['val_acc'][:24])*100, '%\n',
#       'ReLU 64, 25 epoches:\n', 'Test Loss: ',
#       np.mean(history.history['val_loss']), '\n',
#       'Test Accuracy: ', np.mean(history.history['val_acc']) * 100, '%')
#----------------------------------------------------------------------------------------


# ---
# Task 4: *Batch normalization.*
# Build the `new_a_model` again and add batch normalization layers.
# How does it impact your results?*
# ---

keras.backend.clear_session()

#--------------------------Impelment your code here:-------------------------------------
new_activation = 'tanh'

dropout = 0.2
new_a_model = Sequential(name='new_a_model')
new_a_model.add(Flatten(input_shape=[32, 32, 1]))
new_a_model.add(Dense(300, kernel_initializer='he_normal', activation=new_activation))
new_a_model.add(BatchNormalization())
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(150, kernel_initializer="he_normal", activation=new_activation))
new_a_model.add(BatchNormalization())
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(4, activation='softmax'))
#---------------------------------------------------------------------------------------

batch_size = 32
epochs = 50

# Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)
# Compile the network
# Preforming the training by using fit
#--------------------------Impelment your code here:-------------------------------------
# compiling
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

# saving initial weights
if not ("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "init_weights_act_32_bnorm.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved initialized model at %s ' % model_path)

# training the model
history_4 = new_a_model.fit(BaseX_train, BaseY_train,
                         batch_size=batch_size, epochs=epochs,
                         verbose=2,
                         validation_data=(X_test, Y_test))

print(history_4.history.keys())
# accuracy and loss (based on tutorial 11)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history_4.history['acc'])
axs[0].plot(history_4.history['val_acc'])
axs[0].set_title('new_a model accuracy - 32 batch size, with batch norm')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='lower right')


axs[1].plot(history_4.history['loss'])
axs[1].plot(history_4.history['val_loss'])
axs[1].set_title('new_a model loss - 32 batch size, with batch norm')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

# save the model (final weights)
model_name = "final_weights_act_32_bnorm.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved trained model at %s ' % model_path)

# evaluating
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss is {:.2f} ".format(loss_and_metrics[0]))
print("Test Accuracy is {:.2f} %".format(100*loss_and_metrics[1]))
#----------------------------------------------------------------------------------------


# ### PART 2: Convolutional Neural Network (CNN)
# ------------------------------------------------------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *2D CNN.* 
# 
# Have a look at the model below and answer the following:
# * How many layers does it have?
# * How many filter in each layer?
# * Would the number of parmaters be similar to a fully connected NN?
# * Is this specific NN performing regularization?
# 
# ---

print('hi')
def get_net(input_shape,drop,dropRate,reg):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:
    model.add(Dense(512, activation='elu',name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model

# solution of last section: rebuild the function.
def get_net_new(input_shape, drop, dropRate, reg, filters_vec):
    # Defining the network architecture:
    k = 0
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Conv2D(filters=filters_vec[k], kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_1',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    k += 1
    model.add(Conv2D(filters=filters_vec[k], kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_2',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    k += 1
    model.add(Conv2D(filters=filters_vec[k], kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_3',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    k += 1
    model.add(Conv2D(filters=filters_vec[k], kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_4',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    k += 1
    model.add(Conv2D(filters=filters_vec[k], kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_5',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Fully connected network tail:
    model.add(Dense(512, activation='elu', name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu', name='FCN_2'))
    model.add(Dense(4, activation='softmax', name='FCN_3'))
    model.summary()
    return model


input_shape = (32, 32, 1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
NNet = get_net(input_shape,drop,dropRate,reg)



from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.callbacks import *

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network: 
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
# Checkpath = os.getcwd()
# Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)


#Preforming the training by using fit 
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet.save(model_fn)

# NNet.load_weights('Weights_1.h5')

results = NNet.evaluate(X_test, Y_test)
print('test loss, test acc:', results)


# ---
# Task 2: *Number of filters*
# 
# Rebuild the function `get_net` to have as an input argument a list of number of filters in each layers,
# i.e. for the CNN defined above the input should have been `[64, 128, 128, 256, 256]`.
# Now train the model with the number of filters reduced by half.
# What were the results.
# ---

#--------------------------Impelment your code here:-------------------------------------
# function is rebuild above.
orig_filters_vec = [64, 128, 128, 256, 256]
filters_vec = [int(x/2) for x in orig_filters_vec]
NNet = get_net_new(input_shape, drop, dropRate, reg, filters_vec)
#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)
#Compile the network:
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Preforming the training by using fit
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)

results = NNet.evaluate(X_test, Y_test)
print('test loss, test acc:', results)

print('Woohoo!')
#----------------------------------------------------------------------------------------


# That's all folks! See you :)
