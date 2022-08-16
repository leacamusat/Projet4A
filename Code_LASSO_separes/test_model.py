"""
code_test
"""

import tensorflow as tf
import numpy as np 
import time 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ann_lasso_train_model import *
#import pywt####pip3 install PyWavelets
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils 
from sklearn.utils import resample
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

######################################################################### 
time_start=time.time()

#arguments used (directory, numero of te test, number of classes)
directory=sys.argv[1]
numero=sys.argv[2]
classe=int(sys.argv[3])

x_train=pd.read_csv(directory+"/train/ass"+numero+".csv", header=None).to_numpy()
x_test=pd.read_csv(directory+"/test/ass"+numero+".csv",header=None).to_numpy()
non_zero_index=np.where(np.count_nonzero(x_train, axis=0)!=0)[0]
x_train_columns_removed=x_train[:,non_zero_index]
x_test_columns_removed=x_test[:,non_zero_index]

print("début sauvegarde")
np.savetxt(directory+"/train/ass"+numero+"_without_zeros.csv", x_train_columns_removed,  delimiter=',')
np.savetxt(directory+"/test/ass"+numero+"_without_zeros.csv", x_test_columns_removed, delimiter=',')
np.savetxt(directory+"/train/"+numero+"non_zero_index.csv", non_zero_index, delimiter=',')
x_train=pd.read_csv(directory+"/train/ass"+numero+"_without_zeros.csv",header=None).iloc[:, :-classe]
n_train=x_train.shape[0]
y_train=pd.read_csv(directory+"/train/ass"+numero+"_without_zeros.csv", header=None).iloc[:, -classe:]
x_test=pd.read_csv(directory+"/test/ass"+numero+"_without_zeros.csv", header=None).iloc[:, :-classe]
n_test=x_train.shape[0]
y_test=pd.read_csv(directory+"/test/ass"+numero+"_without_zeros.csv", header=None).iloc[:, -classe:]

print('Completed loading data')
#########################################################################   
x_train_l2norm=tf.sqrt(tf.reduce_sum(tf.cast(x_train**2,tf.float64),axis=0))
x_train_l2norm=x_train_l2norm.numpy()

n_test=x_test.shape[0]
x_test_rescaled=x_test/np.repeat(x_train_l2norm.reshape(1,p1),n_test,axis=0)
x_test_rescaled=np.concatenate((x_test_rescaled,np.ones([n_test,1])), axis=1)

w1=np.load(directory+"/w1_"+numero+".npy")
w2=np.load(directory+"/w2_"+numero+".npy")
c=np.load(directory+"/c_"+numero+".npy")       
y_test_hat, p_test_hat, mu_2layer_test=test_model_learnt(x_test_rescaled,  w1, w2, classe, c)

####################################################################################
time_end=time.time()
print("It takes " + str(time_end-time_start) + " seconds for ")

y_test=y_test.to_numpy().transpose()

number_class =y_test.shape[0]

###confusion :row-hat,colume-true
confusion_matrix= np.zeros([number_class,number_class])

number=np.sum(y_test,axis=1)
for i in np.arange(0,number_class):
    for j in np.arange(0,number_class):
       confusion_matrix[i,j]=np.sum(y_test_hat[i,np.where(y_test[j,]==1)])

dataset=directory+"_nb_classes="+classe+"_num_"+numero
np.save(directory+"/phat"+numero, p_test_hat)
np.save(directory+"/ytestobserved_"+numero, y_test_hat)
np.save(directory+"/y_test_"+numero, y_test)

accuracy_test=np.sum(np.diag(confusion_matrix))/n_test

with open("./results_ANN",'a') as file_con:
     file_con.write("**********test****************"+"\n")
     file_con.write('time='+ str(time_end-time_start)+"\n")
     file_con.write('confusion_matrix_test='+str(confusion_matrix)+"\n")
     file_con.write('accuracy_test'+str(accuracy_test)+"\n")
     file_con.write("\n")
     
