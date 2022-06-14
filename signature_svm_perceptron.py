#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:15:31 2022

@author: camusat lea
"""


#librairies utilisées 
import tensorflow as tf
import numpy as np
import random

#tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#signature package (attention necessite l'import de pytorch)
import signatory

#gestion de fichiers 
from os import listdir
from os.path import isfile, join

#data processing package 
from sklearn.model_selection import train_test_split

#pytorch
import torch
import torchvision
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def purity(crosstab): #fonction prenant une table de contingence en entrée et calculant la pureté, propre à la classification
    cross=crosstab.to_numpy()
    purity = 0.
    for cluster in range(cross.shape[1]):  # clusters are along columns
        purity += np.max(cross[ cluster,:])
    return purity/np.sum(cross)


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical


####################################SIGNATURE##########################
######################################################################################################3
###chargement des images de l article 
import pandas as pd
from PIL import Image
import sklearn

fichiers1 = [join("./Documents/projet/Negative", f) for f in listdir("./Documents/projet/Negative") if isfile(join("./Documents/projet/Negative", f))][:2000]
fichiers2 = [join("./Documents/projet/Positive", f) for f in listdir("./Documents/projet/Positive") if isfile(join("./Documents/projet/Positive", f))][:2000]
fichiers=fichiers1+fichiers2
labels1=np.ones(len(fichiers1))[:2000]
labels2=np.zeros(len(fichiers2))[:2000]
labels=np.concatenate((labels1,labels2))


channel,stream,patches=torchvision.io.read_image(fichiers[0]).shape


depth=2
summ=0
for i in range (1,depth+1):
    summ+=channel**i
    
    
# Open the image form working directory
height, width, channels=np.array(Image.open(fichiers[0])).shape
images=np.zeros((len(fichiers),channels,height, width), dtype=np.uint8) #tuple
signatures=np.zeros((len(fichiers),height,summ)) #tuple
signatures_flatten_column=np.zeros((len(fichiers),height*summ))
signatures_flatten_row=np.zeros((len(fichiers),height*summ))
l=0
for i in fichiers:
    image =  torchvision.io.read_image(i)
    
    signatures[l,:,:]=(signatory.signature(np.transpose(image/255, (1,2,0)), depth)).numpy()
    
    signatures_flatten_column[l]=signatures[l].flatten('F')
    signatures_flatten_row[l]=signatures[l].flatten('C')
    
    
    l+=1


###################################################################TESTS modèles ML sans réseau de neurones################################3
####objectif de cette partie : récupérer le vecteur flatten obtenu à partir du tableau 2D de la signature, puis appliquer différents algorithme dessus pour éviter l'utilisation d'un perceptron 
### rappel : Les svm linéaires sont équivalents aux NN à une couche (c'est-à-dire les perceptrons)
### avantages de la méthode : pas de poids à apprendre, ont moins tendance à converger vers des minimas globaux, moins soumis à l overifitting 
### inconvénients de la méthode : perte de l'information spatiale 
print("MODELE EN VECTORISANT PAR COLONNE SVM")
X_trainc, X_testc, y_trainc, y_testc = train_test_split(signatures_flatten_column,labels, test_size=0.3, random_state=42)


param=[{"C":[0.4,0.5,0.6,0.8,1,1.4]}] #paramètres du cout, permettant de mettre en oeuvre des marges flexibles : éviter le surapprentissage en permettant des mauvaises classifications de certains points) 
svm= GridSearchCV(SVC(),param,cv=10,n_jobs=-1)

svmOpt=svm.fit(X_trainc, y_trainc)
# paramètre optimal
print("Meilleur score = %f, Meilleur paramètre = %s" % ( svmOpt.best_score_,svmOpt.best_params_))    


# prévision de l'échantillon test
y_chap = svmOpt.predict(X_testc) # Prédiction
table=pd.crosstab(y_chap,y_testc) # Matrice de confusion 
print("Erreurs et performance du modèle SVM pour prédire")
print("-------------------------------------------------------------------")
# erreur sur l'échantillon test
print("erreur sur l'échantillon test=",1-svmOpt.score(X_testc,y_testc))
print("pureté=", purity(table))
print("-------------------------------------------------------------------")
print("table de contingence")
print(table)
print("\n ratio de prédictions correctes par classe :")
print(np.diag(table)/np.sum(table,axis=0))



print("MODELE EN VECTORISANT PAR LIGNE SVM")
X_trainr, X_testr, y_trainr, y_testr = train_test_split(signatures_flatten_row,labels, test_size=0.3, random_state=42)
param=[{"C":[0.4,0.5,0.6,0.8,1,1.4]}] #paramètres du cout, permettant de mettre en oeuvre des marges flexibles : éviter le surapprentissage en permettant des mauvaises classifications de certains points) 
svm= GridSearchCV(SVC(),param,cv=10,n_jobs=-1)
svmOpt=svm.fit(X_trainr, y_trainr)
# paramètre optimal
print("Meilleur score = %f, Meilleur paramètre = %s" % ( svmOpt.best_score_,svmOpt.best_params_))    
# prévision de l'échantillon test
y_chap = svmOpt.predict(X_testr) # Prédiction
table=pd.crosstab(y_chap,y_testr) # Matrice de confusion 
print("Erreurs et performance du modèle SVM pour prédire")
print("-------------------------------------------------------------------")
# erreur sur l'échantillon test
print("erreur sur l'échantillon test=",1-svmOpt.score(X_testr,y_testr))
print("pureté=", purity(table))
print("-------------------------------------------------------------------")
print("table de contingence")
print(table)
print("\n ratio de prédictions correctes par classe :")
print(np.diag(table)/np.sum(table,axis=0))


#######################################################################################################
print("MODELE EN VECTORISANT PAR LIGNE ARBRE CART")
from sklearn.tree import DecisionTreeClassifier
# Optimisation de la profondeur de l'arbre
param=[{"max_depth":list(range(2,10))}]
tree= GridSearchCV(DecisionTreeClassifier(),param,cv=10,n_jobs=-1)
treeOptclass=tree.fit(X_trainc, y_trainc)
# paramètre optimal

# prévision
y_chap = treeOptclass.predict(X_testc)
# matrice de confusion
table=pd.crosstab(y_chap,y_testc)
print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - treeOptclass.best_score_,treeOptclass.best_params_))
print("Erreurs et performance du modèle CART pour prédire ")
print("-------------------------------------------------------------------")
# erreur sur l'échantillon test
print("erreur sur l'échantillon test=",1-treeOptclass.score(X_testc,y_testc))
print("pureté=", purity(table))
print("-------------------------------------------------------------------")
print("table de contingence")
print(table)
print("\n ratio de prédictions correctes par classe :")
print(np.diag(table)/np.sum(table,axis=0))



#######################################################################################################
print("MODELE EN VECTORISANT PAR LIGNE RF")
from sklearn.ensemble import RandomForestClassifier 
param=[{"max_features":list(range(2,10,1)), "n_estimators":np.arange(100,300,100)}]
rf= GridSearchCV(RandomForestClassifier(),param,cv=5,n_jobs=-1)
rfOpt=rf.fit(X_trainc, y_trainc)
# paramètre optimal
print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - rfOpt.best_score_,rfOpt.best_params_))

# prévision
y_chap = rfOpt.predict(X_testc)
# matrice de confusion
table=pd.crosstab(y_chap,y_testc)

print("Erreurs et performance du modèle RF pour prédire la présence/absence de fissures ")
print("-------------------------------------------------------------------")
# erreur sur l'échantillon test
print("erreur sur l'échantillon test=",1-rfOpt.score(X_testc,y_testc))
print("pureté=", purity(table))
print("-------------------------------------------------------------------")
print("table de contingence")
print(table)
print("\n ratio de prédictions correctes par classe :")
print(np.diag(table)/np.sum(table,axis=0))



#############################################################################################
print("MODELE EN VECTORISANT PAR LIGNE BOOSTING ")
from sklearn.ensemble import GradientBoostingClassifier
param=[{"learning_rate":[0.0001+i*0.001 for i in range (0,15)],"max_depth":np.arange(2,6,1)}]#optimisation de max depth et du learning rate
clfOpt = GridSearchCV(GradientBoostingClassifier(n_estimators=75, random_state=0, loss= "deviance"),  param, cv=5).fit(X_trainc,y_trainc)
print(clfOpt.score(X_testc,y_testc))
print("erreur généralisation la plus basse = %f, Meilleur paramètre = %s" % (clfOpt.best_score_,clfOpt.best_params_))


###############################################################################################
print("MODELE EN VECTORISANT PAR LIGNE AVEC PERCEPTRON ")
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,  input_shape=(summ, ), activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_fn,
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          epochs=2,
          verbose=1,
)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)




model = Sequential()
model.add(Dense(512, input_shape=(784, ), activation='relu'))
model.add(Dense(768, activation='relu'))
model.add(Dense(10, activation='softmax'))