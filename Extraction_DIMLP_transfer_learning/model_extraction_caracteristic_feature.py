#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:31:18 2022

@author: camusat
"""
import six
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Dense, Flatten, Softmax, Dropout
from keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
from keras import models
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow_datasets as tfds
# gestion de fichiers
from os import listdir
from os.path import isfile, join

# data processing package
from sklearn.model_selection import train_test_split

# svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

import time
from datetime import datetime

import csv

from sklearn.model_selection import ShuffleSplit

from numpy import asarray
from numpy import savetxt
import cv2


def create_layer_a(x, kernel_size, filters, padding="same", strides=(1, 1), pool_size=(2, 2)):
    x = Conv2D(kernel_size=kernel_size, filters=filters, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=pool_size)(x)
    return x


def create_layer_b(x, kernel_size, filters, padding="same"):
    x = Conv2D(kernel_size=kernel_size, filters=filters, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def create_layer_c(x, n_neurons):
    x = Dense(n_neurons)(x)
    return ReLU()(x)


def create_layer_d(x, n_neurons):
    x = Dense(n_neurons)(x)
    return Softmax()(x)


def CNN_with_transfer_learning(input_size, nb_class, name):

    """
        Uses VGG16 or EfficientNet and realizes fine tuning with weights learnt with imagenet datasets

        Parameters
        ----------
                input_size : int
                   shape (height, width, number of channels)
                nb_class : int
                   number of classes of the model
                name : int 
                    1 or 2 (correspond to the model 1 : VGG16, 2 : EfficientNet)
                
        """
    #réseau codé en fonctionnel
    #utilise EfficientNetB5, attention ne pas lui donner des images renormalisées mais seulement des images dont les valeurs des pixels sont entre 0 et 255.
    
    #récupère la taille de l'input 
    height, width, channel=input_size
    print(height, width, channel)
   
    inputs = Input(shape=input_size)
    
    if name==1:
        base_model=tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", 
        input_shape = input_size,pooling='max')
        
        

    if name==2: 
        base_model=tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights="imagenet", 
        input_shape = input_size,pooling='max', classes=nb_class)
    #“include_top” argument est à False (non téléchargement des poids de la fully connected
    # de manière à pouvoir faire une autre classification dans le cas d'un transfer learning )
        
    base_model.trainable=False
    #on ne réapprend pas les paramètres déjà appris par le modèle téléchargé 

    
    x=base_model.output
    
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    
    #x = Dense(1024, kernel_regularizer = tf.keras.regularizers.l2(l = 0.016),activity_regularizer=tf.keras.regularizers.l1(0.006),
                    #bias_regularizer=tf.keras.regularizers.l1(0.006) ,activation='relu')(x)
    x = Dense(1024 ,activation='relu')(x)
    x=Dropout(rate=.3, seed=123)(x)
    #x =Dense(128, kernel_regularizer = tf.keras.regularizers.l2(l = 0.016),activity_regularizer=tf.keras.regularizers.l1(0.006),
                    #bias_regularizer=tf.keras.regularizers.l1(0.006) ,activation='relu')(x)
    x =Dense(128 ,activation='relu')(x)
    x=Dropout(rate=.45, seed=123)(x)
    
    if nb_class==2:
        activations="sigmoid"
    else :
        activations="softmax"
    output=Dense(nb_class,activation=activations)(x)

    model = Model(inputs=base_model.inputs, outputs=output)
    #attention à bien prendre inputs=base_model.inputs sinon le modèle est perçu par Tensorflow comme discontinu
    print("here")
    model.summary()

    return model

def CNN_with_fine_tuning(input_size, nb_class, name, trainable): #permet de faire du transfer learning en réapprenant certaines couches 
    #réseau codé en fonctionnel
    #utilise EfficientNetB5, attention ne pas lui donner des images renormalisées mais seulement des images dont les valeurs des pixels sont entre 0 et 255.
    
    """
        Uses VGG16 or EfficientNet and realizes fine tuning with weights 
        learnt with imagenet datasets. Weights of specific layers are re-learnt again

        Parameters
        ----------
                input_size : int
                   shape (height, width, number of channels)
                nb_class : int
                   number of classes of the model
                name : int 
                    1 or 2 (correspond to the model 1 : VGG16, 2 : EfficientNet)
                trainable : list
                   indexs of the layers that we want to re-learn
        """

    #récupère la taille de l'input 
    height, width, channel=input_size
    print(height, width, channel)
   
    inputs = Input(shape=input_size)
    model_name='EfficientNetB5'
    if name==1:
        base_model=tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", 
        input_shape = input_size,pooling='max')
    if name==2:
        base_model=tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights="imagenet", 
        input_shape = input_size,pooling='max', classes=nb_class)
    #“include_top” argument est à False (non téléchargement des poids de la fully connected
    # de manière à pouvoir faire une autre classification dans le cas d'un transfer learning )

    #print(base_model.summary())
    
    # Freeze all the layers
    for i, layer in enumerate(base_model.layers):
        print(layer.trainable)
        if i in trainable:
            layer.trainable = True
        else :
            layer.trainable = False
        print(layer.trainable)
        print("---------------------")
    print("coucou--------------------------------------------------------------------------------------------")
# Check the trainable status of the individual layers
    for layer in base_model.layers:
        print(layer, layer.trainable)

    print("line 100")
    x=base_model.output
    print("line 102")
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    print("line 104")
    #x = Dense(1024, kernel_regularizer = tf.keras.regularizers.l2(l = 0.016),activity_regularizer=tf.keras.regularizers.l1(0.006),
                    #bias_regularizer=tf.keras.regularizers.l1(0.006) ,activation='relu')(x)
    x = Dense(1024 ,activation='relu')(x)
    x=Dropout(rate=.3, seed=123)(x)
    #x =Dense(128, kernel_regularizer = tf.keras.regularizers.l2(l = 0.016),activity_regularizer=tf.keras.regularizers.l1(0.006),
                    #bias_regularizer=tf.keras.regularizers.l1(0.006) ,activation='relu')(x)
    x =Dense(128 ,activation='relu')(x)
    x=Dropout(rate=.45, seed=123)(x)
    print("line 106")
    if nb_class==2:
        activations="sigmoid"
    else :
        activations="softmax"
    output=Dense(nb_class,activation=activations)(x)

    


    model = Model(inputs=base_model.inputs, outputs=output)
    #attention à bien prendre inputs=base_model.inputs sinon le modèle est perçu par Tensorflow comme discontinu
    print("here, 199")
    model.summary()

    return model


def create_model(input_size, output_size, n_heads=3):
    print("here")
    inputs = Input(shape=input_size)
    print("here")
    x = create_layer_a(inputs, (11, 11), 96, strides=(4, 4))
    print("here")
    for i in range(2):
        x = create_layer_b(x, (3, 3), 256)

    x = create_layer_a(x, (3, 3), 256)
    print("here")
    for i in range(2):
        x = create_layer_b(x, (3, 3), 384)

    x = create_layer_a(x, (3, 3), 384)

    for i in range(2):
        x = create_layer_b(x, (3, 3), 512)

    #x = create_layer_a(x, (3, 3), 512)
    #x = create_layer_a(x, (3, 3), 512, strides=(2, 2))
    print("here")
    x = Flatten()(x)
    print("here")
    x = create_layer_c(x, 2048)
    print("here")
    outputs = create_layer_d(x, output_size)
    print("here2")
    model = Model(inputs=inputs, outputs=outputs)
    print("here")
    model.summary()
    
    return model


def normalize(x1, x2):
    x1 = x1/255.0
    
    return x1, x2

def preprocess(images, labels):
  return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

def prepa_transfer_learning(img, y):
    """
        reshape image of datasets smaller than 32 (size needed by EfficientNet)

        Parameters
        ----------
                img : images 
                labels : labels
                Ces deux arguments sont passés sous la forme d'un 
         Returns
         ----------
            
        """
    print("IMAGE_SHAPE", img.shape)
    img = tf.image.resize(img,             # resize acc to the input
             [32, 32])
    img=tf.image.grayscale_to_rgb(img)
    print("IMAGE_SHAPE_AFTER", img.shape)
    y=tf.one_hot(y, 10)

    return img, y


if __name__ == "__main__":
    nb_class=int(sys.argv[4])
    if (len(sys.argv)<4):
        print("--------------------------------------------------------------------------------")
        print("argument 1 : 0=model_normal,1=Transferlearning, 2: Fine_tuning")
        print("argument 2 : dataset = 1. : melanoma 2.: mnist")
        print("argument 3 : 1 : vgg16, 2 : EfficientNet")
        print("argument 4 : rentrer le nombre de classes que l'on veut prédire")
    transfer_learning=int(sys.argv[1])
    #bool(int(input('Transfer learning ? (0=model_normal,1=Transferlearning, 2: Fine_tuning):')))
    #dataset_name=int(input('Entrez le dataset ex : 1 melanoma :'))
    dataset_name=int(sys.argv[2])
    #dataset_name=1
    


    if (dataset_name == 1): 
    #principe de cette fonction : récupère les données dans le répertoire train, les labellise automatiquement car dans train il y a deux repertoires 
        ds_train = tf.keras.utils.image_dataset_from_directory("./melanoma/train", label_mode="categorical", batch_size=32, image_size=(224, 224))
        ds_test = tf.keras.utils.image_dataset_from_directory("./melanoma/test", label_mode="categorical", batch_size=32, image_size=(224, 224))

    #ds_train = ds_train.map(lambda x1, x2: normalize(x1, x2)) #renormalisation
    #ds_test = ds_test.map(lambda x1, x2: normalize(x1, x2)) #renormalisation
    #ds_valid, ds_train=tfds.split(ds_train, left_size=0.25)

        total =ds_train

        ds_train = total.enumerate() \
                    .filter(lambda x,y: x % 4!= 0) \
                    .map(lambda x,y: y)

        valid_dataset = total.enumerate() \
                    .filter(lambda x,y: x % 4 == 0) \
                    .map(lambda x,y: y)


    if (dataset_name==2):
        ds_train = tfds.load('mnist', split='train[:70%]',  as_supervised=True, batch_size=32)
        
        valid_dataset = tfds.load('mnist', split='train[70%:80%]',
                       as_supervised=True, batch_size=32)
        ds_test=tfds.load('mnist', split='train[-20%:]',
                       as_supervised=True, batch_size=32)
        
        ds_train=ds_train.map(prepa_transfer_learning)
        ds_test=ds_test.map(prepa_transfer_learning)
        valid_dataset=valid_dataset.map(prepa_transfer_learning)
        #le as_supervised=True permet de récupérer comme l'ensemble de x et de y

      
    callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    )
]
    #name= int(input('Entrez le modèle que vous souhaitez utiliser en transfer learning (entre guillemets) : 1 vgg | 2 Net :'))
    name=int(sys.argv[3])
    if name ==1:
        ds_train=ds_train.map(preprocess)
        valid_dataset=valid_dataset.map(preprocess)
        ds_test=ds_test.map(preprocess) #permet de préprocesser les images tels que c'est généralement fait pour vgg16 

    if dataset_name==1:
        shape=(224,224,3)
    if dataset_name==2:
        shape=(32, 32, 3)
    if transfer_learning==1:
        model= CNN_with_transfer_learning(shape, nb_class, name)
    elif transfer_learning==2:
        model=CNN_with_fine_tuning(shape, nb_class,  name, [17])
    else :
        model  = create_model(shape, nb_class)
    opt= tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics="accuracy")


    history=model.fit(ds_train, epochs=100, batch_size=256, validation_data=valid_dataset, callbacks=callbacks)
    print("ok")
    model.evaluate(ds_test)
    print("modèle évalué")

    
    # si on fait sparse_categorical_crossentropy, ne convertit pas en 0 0 1 les classes mais prend directement 3, 
    # et on n a pas besoin de modifier y. 


 
    ensemble_image_label=ds_test.skip(1).take(1)

    for images, labels in ensemble_image_label:  # only take first element of dataset
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
    
    layer_outputs = [layer.output for layer in model.layers[17:18]] # Extracts the outputs of the top 12 layers
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name) # Names of the layers
    print("extraction de la couche", layer_names[17])
    fig=plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('./loss.png', dpi=fig.dpi)
    
    fig=plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    fig.savefig('./accuracy.png', dpi=fig.dpi)



    
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activations_test = activation_model.predict(ds_test) 
    activations_train=activation_model.predict(ds_train) 


    #activations_train = activation_model.predict(ds_train) 
    print("activation", np.shape(activations_train))
    np.save("activations_train.txt", activations_train)
    np.save("activations_test.txt", activations_test)
    #activations_test=activation_model.predict(ds_test) 


    #recupere les differents labels de l'ensemble test
    i=0
    for images, labels in ds_test:  
        if i==0:
            i+=1
            labels_total_test=labels.numpy()
        else :
            numpy_images = images.numpy()
            numpy_labels = labels.numpy()
            labels_total_test=np.concatenate((labels_total_test, numpy_labels))   
    
    #recupere les différents labels de l'ensemble train
    i=0
    for images, labels in ds_train:  
        if i==0:
            i+=1
            labels_total_train=labels.numpy()
        else :
            numpy_labels = labels.numpy()
            labels_total_train=np.concatenate((labels_total_train, numpy_labels)) 

        


    np.savetxt("./labels_test.csv", labels_total_test , delimiter=',')
    np.savetxt("./labels_train.csv", labels_total_train , delimiter=',')



