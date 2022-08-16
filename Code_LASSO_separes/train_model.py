"""
train_model
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

#####treatment when there are columns with zeros, we remove them
directory=sys.argv[1]
numero=sys.argv[2]
classe=int(sys.argv[3])

x_train=pd.read_csv(directory+"/train/ass"+numero+".csv", header=None).to_numpy()
non_zero_index=np.where(np.count_nonzero(x_train, axis=0)!=0)[0]
x_train_columns_removed=x_train[:,non_zero_index]

print("début sauvegarde")
np.savetxt(directory+"/train/ass"+numero+"_without_zeros.csv", x_train_columns_removed,  delimiter=',')
print("fin sauvegarde")
np.savetxt(directory+"/train/"+numero+"non_zero_index.csv", non_zero_index, delimiter=',')

x_train=pd.read_csv(directory+"/train/ass"+numero+"_without_zeros.csv",header=None).iloc[:, :-classe]
n_train=x_train.shape[0]
y_train=pd.read_csv(directory+"/train/ass"+numero+"_without_zeros.csv", header=None).iloc[:, -classe:]

x_train=x_train.values.astype(float)

y_train=y_train.values.astype('float64')

print('Completed loading data')
#########################################################################   

###should be compile
n_train,p1=x_train.shape
x_train_l2norm=tf.sqrt(tf.reduce_sum(tf.cast(x_train**2,tf.float64),axis=0))
x_train_l2norm=x_train_l2norm.numpy()
x_train_rescaled=x_train/np.repeat(x_train_l2norm.reshape(1,p1),n_train,axis=0)
x_train_rescaled=np.concatenate((x_train_rescaled,np.ones([n_train,1])), axis=1)


y_train=y_train.transpose()
hat_p_training = tf.reduce_mean(tf.cast(y_train,tf.float64),axis=1)
nSample=100
lambda_qut=np.load(directory+"/_"+str(numero)+"_lambda_qut.npy")
print("lambda_qut=", lambda_qut)
p2=20
num_rep = 1
learningRate_list=[0.01, 0.001]
iniscale=0.001

w1o,w2o,co,p_hat_o,y_hat_o,cost_o,lro,needles_index_hat, neuron_index, p_hat_o=computeResults(x_train_rescaled,y_train,learningRate_list,iniscale,lambda_qut,p2) 


np.save(directory+"/act(w1_x_train)"+numero, activation_function(tf.matmul(w1o,tf.transpose(x_train_rescaled))))
np.save(directory+"/w1_"+numero, w1o)
np.save(directory+"/w2_"+numero, w2o)
np.save(directory+"/c_"+numero, co)

####################################################################################
time_end=time.time()
print("It takes " + str(time_end-time_start) + " seconds for "+str(num_rep)+" times." )

y_train=y_train.to_numpy()
number_class =y_train.shape[0]
confusion_matrix_train= np.zeros([number_class,number_class])

number=np.sum(y_train,axis=1)
for i in np.arange(0,number_class):
    for j in np.arange(0,number_class):
       confusion_matrix_train[i,j]=np.sum(y_hat_o[i,np.where(y_train[j,]==1)])
accuracy_train=np.sum(np.diag(confusion_matrix_train))/n_train

dataset=directory+"_nb_classes="+classe+"_num_"+numero

with open("./results_ANN",'a') as file_con:
     file_con.write("\n")
     file_con.write('dataset='+dataset+"\n")
     file_con.write('time='+ str(time_end-time_start)+"\n")
     file_con.write('lambda_qut='+str(lambda_qut)+"\n")
     file_con.write('cost_o='+str(cost_o)+"\n")
     file_con.write('accuracy_on_train='+str(accuracy_train)+"\n")
     file_con.write('confusion_matrix_train='+str(confusion_matrix_train)+"\n")
     file_con.write("shape_needles="+str(np.shape(needles_index_hat)[1])+"\n")
     file_con.write("needles_index="+str(needles_index_hat)+"\n")
     file_con.write("neurons_index="+str(neuron_index)+"\n")
     file_con.write("filters_index"+str( np.ceil(np.array(needles_index_hat)/float(p1)).astype(int)))
     file_con.write("\n")