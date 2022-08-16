"""
calculation of lambda qut
"""

import tensorflow as tf
import numpy as np 
import time 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ann_lasso_classification_2layer import *
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


#########################################################################################################################

x_train=x_train.values.astype(float)
y_train=y_train.astype('float64')

print('Completed loading data')
#########################################################################   
###should be compile
n_train,p1=x_train.shape
x_train_l2norm=tf.sqrt(tf.reduce_sum(tf.cast(x_train**2,tf.float64),axis=0))
x_train_l2norm=x_train_l2norm.numpy()
print("norm", np.where(x_train_l2norm==0))
x_train_rescaled=x_train/np.repeat(x_train_l2norm.reshape(1,p1),n_train,axis=0)
x_train_rescaled=np.concatenate((x_train_rescaled,np.ones([n_train,1])), axis=1)


y_train=y_train.transpose()

hat_p_training = tf.reduce_mean(tf.cast(y_train,tf.float64),axis=1)
nSample=100

lambda_qut=lambda_qut_sann_classification(tf.nn.l2_normalize(x_train,axis=0),hat_p_training,nSample=nSample,miniBatchSize=20,alpha=0.05,option='quantile')
np.save(directory+"/_"+str(numero)+"_lambda_qut", lambda_qut)

p2=20
num_rep = 1
learningRate_list=[0.01, 0.001]
iniscale=0.001

####################################################################################
time_end=time.time()
print("It takes " + str(time_end-time_start) + " seconds for "+str(num_rep)+" times." )

