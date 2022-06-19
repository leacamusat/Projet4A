#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 08:29:55 2022

@author: camusat
"""


# librairies utilisées
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



def purity(
    crosstab,
):  # fonction prenant une table de contingence en entrée et calculant la pureté, propre à la classification
    cross = crosstab.to_numpy()
    purity = 0.0
    for cluster in range(cross.shape[1]):  # clusters are along columns
        purity += np.max(cross[cluster, :])
    return purity / np.sum(cross)


def svm(optimisation, X_trainc, X_testc, y_trainc, y_testc, nom_fichier, param=[{"C": [0.4, 0.5, 0.6, 0.8, 1, 1.4]}]):
    if optimisation==True:
        print("MODELE EN VECTORISANT PAR COLONNE SVM AVEC OPTIMISATION DU COUT \n")
        param = [
            {"C": [0.4, 0.5, 0.6, 0.8, 1, 1.4]}
        ]  # paramètres du cout, permettant de mettre en oeuvre des marges flexibles : éviter le surapprentissage en permettant des mauvaises classifications de certains points)
        svm = GridSearchCV(SVC(), param, cv=10, n_jobs=-1)
        # entrainement du modèle
        start = time.time()
        svmOpt = svm.fit(X_trainc, y_trainc)
        end = time.time()
        elapsed = end - start
        
    
        # paramètre optimal
        print(
            "Meilleur score = %f, Meilleur paramètre = %s"
            % (svmOpt.best_score_, svmOpt.best_params_)
        )
        C=svmOpt.best_params_
    
    
        # prévision de l'échantillon test
        y_chap = svmOpt.predict(X_testc)  # Prédiction
        table = pd.crosstab(y_chap, y_testc)  # Matrice de confusion
        print("Erreurs et performance du modèle SVM pour prédire")
        print("-------------------------------------------------------------------")
        # erreur sur l'échantillon test
        print("erreur sur l'échantillon test=", 1 - svmOpt.score(X_testc, y_testc))
        print("pureté=", purity(table))
        print("-------------------------------------------------------------------")
        print("table de contingence")
        print(table)
        print("\n ratio de prédictions correctes par classe :")
        print(np.diag(table) / np.sum(table, axis=0))
    
        print(f"Temps d'apprentissage : {elapsed:.2}ms")
    
        table = pd.crosstab(y_chap, y_testc)
        print("pureté=", purity(table))
    
    
        print(
            "F1-score=",
            f1_score(
                y_chap, y_testc, average="macro"
            ),
        )
    else :   
        param = [
            {"C": [0.4, 0.5, 0.6, 0.8, 1, 1.4]}
        ]  # paramètres du cout, permettant de mettre en oeuvre des marges flexibles : éviter le surapprentissage en permettant des mauvaises classifications de certains points)
        svm1 = SVC()
    
    
        # entrainement du modèle
        start = time.time()
    
       
        
        
    
        svm = svm1.fit(X_trainc, y_trainc)
        
        end = time.time()
    
        elapsed = end - start
        
    
        # paramètre optimal
       
        C=1
    
    
        # prévision de l'échantillon test
        y_chap = svm.predict(X_testc)  # Prédiction
        table = pd.crosstab(y_chap, y_testc)  # Matrice de confusion
        print("Erreurs et performance du modèle SVM pour prédire")
        print("-------------------------------------------------------------------")
        # erreur sur l'échantillon test
        print("erreur sur l'échantillon test=", 1 - svm.score(X_testc, y_testc))
        print("pureté=", purity(table))
        print("-------------------------------------------------------------------")
        print("table de contingence")
        print(table)
        print("\n ratio de prédictions correctes par classe :")
        print(np.diag(table) / np.sum(table, axis=0))
    
        print(f"Temps d'apprentissage : {elapsed:.2}ms")
    
        table = pd.crosstab(y_chap, y_testc)
        print("pureté=", purity(table))
    
        F1_score=f1_score(y_chap, y_testc, average="macro")
        print("F1-score=", F1_score)
        
    
    return purity(table), F1_score, 1 - svm.score(X_testc, y_testc), elapsed, C



def OneDCNN(optimisation, X_train, X_test, y_train, y_test, height, width, channel, nom_fichier, depth, epoch, batch):
    # structure du modèle conv 1D
    summ = 0
    for i in range(1, depth + 1):
        summ += channel ** i
    modelconv1D = tf.keras.models.Sequential()  # réseau vide
    modelconv1D.add(tf.keras.layers.Conv1D(filters=32, kernel_size=(3), input_shape=(height, summ), activation="relu"))
    modelconv1D.add(tf.keras.layers.MaxPooling1D(pool_size=3))
    modelconv1D.add(tf.keras.layers.Conv1D(filters=64, kernel_size=(3), activation="relu"))
    
    # Ajout de la première couche de pooling
    modelconv1D.add(tf.keras.layers.MaxPooling1D(pool_size=3))

    modelconv1D.add(tf.keras.layers.Flatten())  # Conversion
    modelconv1D.add(tf.keras.layers.Dense(50, activation="relu"))
    modelconv1D.add(tf.keras.layers.Dense(2, activation="softmax"))


    modelconv1D.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    
    start = time.time()
    history = modelconv1D.fit(
        X_train, to_categorical(y_train), epoch, batch
    )
    modelconv1D.evaluate(X_test, to_categorical(y_test))


    end = time.time()

    elapsed = end - start

    print("résumé des couches du modèle")
    stringlist = []
    modelconv1D.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)
    print(type(short_model_summary))

    print(f"Temps d'apprentissage : {elapsed:.2}ms")

    table = pd.crosstab(np.argmax(modelconv1D.predict(X_test), axis=1), y_test)
    print("pureté=", purity(table))

    F1_score=f1_score(np.argmax(modelconv1D.predict(X_test), axis=1), y_test, average="macro")
    print("F1-score=", F1_score)


    
    modelconv1D.save(nom_fichier[:-6]+"poids_CNN1D_depth="+str(depth))

    #history.history['loss']
    
    return modelconv1D.evaluate(X_test, to_categorical(y_test))[1], F1_score, modelconv1D.evaluate(X_test, to_categorical(y_test))[0], elapsed, modelconv1D.count_params()


