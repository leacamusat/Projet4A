#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:15:31 2022
@author: camusat lea
"""
# libraiies used 
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import sklearn
import os

# tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical


# signature package (attention necessite l'import de pytorch)
import signatory

# gestion de fichiers
from os import listdir
from os.path import isfile, join

# data processing package
from sklearn.model_selection import train_test_split

# pytorch
import torch
import torchvision
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

import time
from datetime import datetime

import csv

import methods_on_signature as meth

from sklearn.model_selection import ShuffleSplit

from numpy import asarray
from numpy import savetxt




####################################SIGNATURE##########################
######################################################################################################3
###image download for cracks 
fichiers1 = [
    join("./Negative", f)
    for f in listdir("./Negative")
    if isfile(join("./Negative", f))
][:2000]
fichiers2 = [
    join("./Positive", f)
    for f in listdir("./Positive")
    if isfile(join("./Positive", f))
][:2000]
fichiers = fichiers1 + fichiers2
labels1 = np.ones(len(fichiers1))[:2000]
labels2 = np.zeros(len(fichiers2))[:2000]
labels = np.concatenate((labels1, labels2))


channel, stream, patches = torchvision.io.read_image(fichiers[0]).shape

# # Model / data parameters
# num_classes = 10
# input_shape = (28, 28, 1)

# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

###learner is a vector which contains the different methods which we have tested 1:SVM, 2:CART, 3:
    
## signature is a function with several inputs : the dataset with a size of 
learner=[meth.svm, meth.OneDCNN ]
namelearner=["SVM", "CNN1D"]
def signature(dataset, dataname, labels, optimisation, number_MC, depth, numlearner, test_ratio, epoch, batch, filesname=True):
    
    if not os.path.exists("./"+"tests"+dataname+"_"+str(datetime.now())[:-16]):
        os.makedirs("./"+"tests/tests_"+dataname+"_"+str(datetime.now())[:-16], exist_ok=True)
    
        nom_fichier="./"+"tests/tests_"+dataname+"_"+str(datetime.now())[:-16]+"/"+str(datetime.now())[:-16]+"_"+str(datetime.now())[-15:-13]+"_"+str(datetime.now())[-12:-10]+".csv" #création du fichier test avec date et heure
        nom_fichier_tensor="./"+"tests/tests_tensor_signature"+dataname+"_"+str(datetime.now())[:-16]+"/"+str(datetime.now())[:-16]+"_"+str(datetime.now())[-15:-13]+"_"+str(datetime.now())[-12:-10]+".csv" #création du fichier test avec date et heure
    
    else : 
        nom_fichier="./"+"Documents/projet/tests/tests_"+dataname+"_"+str(datetime.now())[:-16]+"/"+str(datetime.now())[:-16]+"_"+str(datetime.now())[-15:-13]+"_"+str(datetime.now())[-12:-10]+str(depth)+".csv" 
        nom_fichier_tensor="./"+"Documents/projet/tests/tests_tensor_signature"+dataname+"_"+str(datetime.now())[:-16]+"/"+str(datetime.now())[:-16]+"_"+str(datetime.now())[-15:-13]+"_"+str(datetime.now())[-12:-10]+".csv" 
    
    
    with open(nom_fichier,'w',newline='') as fichiercsv:
       writer=csv.writer(fichiercsv)
       writer.writerow( ["TESTS,depth="+str( depth), " ", " ", " ", " ", " "])
    
    
    summ = 0
    for i in range(1, depth + 1):
        summ += channel ** i
    key_list = np.sort(np.unique(labels))
    value_list = np.arange(0, len(np.unique(labels))) #number of labels

    dict_from_list = dict(zip(key_list, value_list))    
    # Open the image form working directory
    labels_array=np.zeros((len(labels), len(np.unique(labels))))
    for i in range (len(labels)):
        labels_array[i, dict_from_list[labels[i]]]=1
    
    height, width, channels = np.array(Image.open(dataset[0])).shape
    
    
    images = np.zeros((len(dataset),  height, width,channels,), dtype=np.uint8)  # tuple
    signatures = np.zeros((len(dataset), height, summ))  # tuple
    signatures_flatten_column = np.zeros((len(dataset), height * summ))
    signatures_flatten_row = np.zeros((len(dataset), height * summ))
    
    l = 0
    
    for i in fichiers:
        image = torchvision.io.read_image(i)
        signatures[l, :, :] = (
            signatory.signature(np.transpose(image / 255, (1, 2, 0)), depth)).numpy()
        images[l]=np.transpose(image, (1, 2, 0))
        signatures_flatten_column[l] = signatures[l].flatten("F")
        signatures_flatten_row[l] = signatures[l].flatten("C")
    
        l += 1
    print(signatures_flatten_row.shape)
    # save to csv file
    savetxt("./tests/"+dataname+'_depth='+str(depth)+'.csv', np.transpose(signatures_flatten_row), delimiter=',')
    savetxt("./tests/"+dataname+'_depth='+str(depth)+'_labels.csv', labels_array, delimiter=',')
    #Then, to reload:
    #df = pd.read_csv("testfile")
    
    accuracy_vec=np.zeros((number_MC, len(numlearner)))
    err_vec=np.zeros((number_MC, len(numlearner)))
    F1_vec=np.zeros((len(numlearner), number_MC))
    
    
    X_trainc, X_testc, y_trainc, y_testc = X_trainc, X_testc, y_trainc, y_testc = train_test_split(
        signatures_flatten_column, labels, test_size=0.3, random_state=42)
    
    
    
    rs = ShuffleSplit(n_splits=number_MC, test_size=test_ratio, random_state=0)
    trainlistMC=[]
    testlistMC=[]
    count=0
    for train_data, test_data in rs.split(labels):
        trainlistMC.append(train_data)
        testlistMC.append(test_data)
    for j in numlearner : 
        
        for MC in range (number_MC):
            X_trainc=signatures_flatten_column[trainlistMC[MC]]
            y_trainc=labels[trainlistMC[MC]]
            y_testc=labels[testlistMC[MC]]
            X_testc=signatures_flatten_column[testlistMC[MC]]
            
            X_train= signatures[trainlistMC[MC]]
            X_test=signatures[testlistMC[MC]]
            with open(nom_fichier,'a',newline='') as fichiercsv:
                writer=csv.writer(fichiercsv)
                if (MC==0 and count==0):
                    writer.writerow( ['depth:'+str(depth), 'accuracy', 'F1-score','test_error', 'learning_time(ms)', 'number_of_parameters_(for NN)'])
                
                
            if (j==0 or j==2 or j==3 or j==4):
                
                accuracy, F1, err, temps, C=learner[j](optimisation,X_trainc, X_testc, y_trainc, y_testc, nom_fichier)
                
                print(err_vec.shape)
                print('MC, count', MC, count)
                err_vec[MC,count]=err
                
                
                
                with open(nom_fichier,'a',newline='') as fichiercsv:
                    writer=csv.writer(fichiercsv)
                    
                    writer.writerow(["SVM_POST_SIGNATURE_C="+str(C)+'_MC:'+str(MC+1)+"/"+ str(number_MC), str(accuracy), str(F1), str(err), str(temps), "NA"])
            
                    
            if (j ==1):
                accuracy, F1, temps, err, param=learner[j](optimisation, X_train, X_test, y_trainc, y_testc, height, width, channels,nom_fichier, depth, epoch, batch)
                with open(nom_fichier,'a',newline='') as fichiercsv:
                    writer=csv.writer(fichiercsv)
                    
                    
                    writer.writerow(["CNN_1D_POST_SIGNATURE_MC:"+str(MC+1)+"/"+ str(number_MC), str(accuracy), str(F1), str(err), str(temps), str(param)])
                err_vec[MC,count]=err
                   
                
        count=count+1
        
    fig = plt.figure()
    fig.suptitle('depth='+str(depth)+ 'Monte Carlo, MC='+str(MC), fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    ax.boxplot(err_vec)
    
    ax.set_title('boxplot des erreurs')
    legend=[]
    for n in range (len(numlearner)):
        legend.append(namelearner[n])
    ax.set_xlabel(legend)
    plt.savefig("./tests/boxplot_des_erreurs_depth="+str(depth)+".png",dpi=300)

    plt.show()     
    return 0


#call of the function 
signature(fichiers, "cracks", labels, False, 3, 3, [0,1], 0.3, 300,300)


