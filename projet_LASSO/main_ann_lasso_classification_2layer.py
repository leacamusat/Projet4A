"""
ann_lasso_classification_2layer
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


# ###########################################################################
# x=pd.read_csv('./cifar10_depth=3.csv',header=None)
# n_train=x.shape[0]
# y=pd.read_csv('./cifar10_depth=3_labels.csv',header=None)

# x=pd.read_csv('./Cracks_depth=3.csv',header=None)
# n_train=x.shape[0]
# y=pd.read_csv('./Cracks_depth=3_labels.csv',header=None)

# x=pd.read_csv('./ECG_depth=2.csv',header=None)
# n_train=x.shape[0]
# y=pd.read_csv('./ECG_depth=2_labels.csv',header=None)
#################################################
# x=pd.read_csv('./melanoma_depth=2.csv',header=None)
# n_train=x.shape[0]
# y=pd.read_csv('./melanoma_depth=2_labels.csv',header=None)

# all_dataset_concat= pd.concat([x.transpose(),y], axis=1)

# train=all_dataset_concat.sample(frac=0.6,random_state=200) #random state is a seed value
# test=all_dataset_concat.drop(train.index)


# x_train=train.iloc[:,:n_train]
# x_test=test.iloc[:,:n_train]
# y_test=test.iloc[:, n_train:]
# y_train=train.iloc[:, n_train:]
####################################################

# x_test=pd.read_csv('./output4/test_convol.csv',header=None).iloc[:, :-2]
# n_test=x_test.shape[0]
# y_test=pd.read_csv('./output4/test_convol_labels.csv',header=None)

# x_train=pd.read_csv('./output4/train_convol.csv',header=None).iloc[:, :-2]
# n_train=x_train.shape[0]
# y_train=pd.read_csv('./output4/train_convol_labels.csv',header=None)

# x_val=pd.read_csv('./output4/val_convol.csv',header=None).iloc[:, :-2]
# n_val=x_val.shape[0]
# y_val=pd.read_csv('./output4/val_convol_labels.csv',header=None)

###################################################

# x_test=pd.read_csv('./output3/maxpool/val_maxpool.csv',header=None).iloc[:, :-2]
# n_test=x_test.shape[0]
# y_test=pd.read_csv('./output3/maxpool/val_maxpool.csv',header=None).iloc[:, -2:]

# x_train=pd.read_csv('./output3/maxpool/train_maxpool.csv',header=None).iloc[:, :-2]
# n_train=x_train.shape[0]
# y_train=pd.read_csv('./output3/train_convol_labels.csv',header=None).iloc[:, -2:]

# x_val=pd.read_csv('./output3/maxpool/val_maxpool.csv',header=None).iloc[:, :-2]
# n_val=x_val.shape[0]
# y_val=pd.read_csv('./output3/maxpool/val_maxpool.csv',header=None).iloc[:, -2:]

# x_test=pd.read_csv('outputMNIST.csv',header=None).iloc[:, :-10]
# n_test=x_test.shape[0]
# y_test=pd.read_csv('outputMNIST.csv',header=None).iloc[:, -10:]



# x_train=pd.read_csv('outputMNISTval.csv',header=None).iloc[:, :-10]
# n_train=x_train.shape[0]
# y_train=pd.read_csv('outputMNISTval.csv',header=None).iloc[:, -10:]

# ind=x_train.index[x_train==0.any(axis=0)]
# print(ind, "ind")
# x_train.drop(ind,1,inplace=True)
# print("1")
# x_test.drop(ind,1,inplace=True)
# print("soleil")

# x_test=pd.read_csv('./MNIST/test/ass0.csv',header=None).iloc[:, :-2]
# n_test=x_test.shape[0]
# y_test=pd.read_csv('./MNIST/test/ass0.csv',header=None).iloc[:, -2:]


# x_train=pd.read_csv('./MNIST/train/ass0.csv',header=None).iloc[:, :-2]
# n_train=x_train.shape[0]
# y_train=pd.read_csv('./MNIST/train/ass0.csv',header=None).iloc[:, -2:]


#####treatment when there are columns with zeros, we remove them
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
print("fin sauvegarde")
np.savetxt(directory+"/test/ass"+numero+"_without_zeros.csv", x_test_columns_removed, delimiter=',')
np.savetxt(directory+"/train/"+numero+"non_zero_index.csv", non_zero_index, delimiter=',')

x_train=pd.read_csv(directory+"/train/ass"+numero+"_without_zeros.csv",header=None).iloc[:, :-classe]
n_train=x_train.shape[0]
y_train=pd.read_csv(directory+"/train/ass"+numero+"_without_zeros.csv", header=None).iloc[:, -classe:]


x_test=pd.read_csv(directory+"/test/ass"+numero+"_without_zeros.csv", header=None).iloc[:, :-classe]
n_test=x_train.shape[0]
y_test=pd.read_csv(directory+"/test/ass"+numero+"_without_zeros.csv", header=None).iloc[:, -classe:]
#########################################################################################################################
# x_test=pd.read_csv('outputMNIST.csv',header=None).iloc[:, :-10]
# n_test=x_test.shape[0]
# y_test=pd.read_csv('outputMNIST.csv',header=None).iloc[:, -10:]


# x_val=pd.read_csv('./output3/maxpool/val_maxpool.csv',header=None).iloc[:, :-2]
# n_val=x_val.shape[0]
# y_val=pd.read_csv('./output3/maxpool/val_maxpool.csv',header=None).iloc[:, -2:]

x_train=x_train.values.astype(float)
x_test=x_test.values.astype(float)



y_train=y_train.astype('float64')
y_test=y_test.astype('float64')


print('Completed loading data')
#########################################################################   
# filename_confusion=dir_name_results+'confusion_matrix/'+method_name+'_'+data_name+'_confusion_matrix.txt'
# filename_accuracy=dir_name_results+'accuracy/'+method_name+'_'+data_name+'_accuracy.txt'
# filename_needles=dir_name_results+'needles_index/'+method_name+'_'+data_name+'_needles_index.txt'
# filename_num_needles=dir_name_results+'num_needles/'+method_name+'_'+data_name+'_num_needles.txt'

###should be compile
n_train,p1=x_train.shape
x_train_l2norm=tf.sqrt(tf.reduce_sum(tf.cast(x_train**2,tf.float64),axis=0))
x_train_l2norm=x_train_l2norm.numpy()
print("norm", np.where(x_train_l2norm==0))
x_train_rescaled=x_train/np.repeat(x_train_l2norm.reshape(1,p1),n_train,axis=0)
x_train_rescaled=np.concatenate((x_train_rescaled,np.ones([n_train,1])), axis=1)
n_test=x_test.shape[0]
x_test_rescaled=x_test/np.repeat(x_train_l2norm.reshape(1,p1),n_test,axis=0)
x_test_rescaled=np.concatenate((x_test_rescaled,np.ones([n_test,1])), axis=1)

y_train=y_train.transpose()
y_test=y_test.transpose()
hat_p_training = tf.reduce_mean(tf.cast(y_train,tf.float64),axis=1)
nSample=100
lambda_qut=lambda_qut_sann_classification(tf.nn.l2_normalize(x_train,axis=0),hat_p_training,nSample=nSample,miniBatchSize=20,alpha=0.05,option='quantile')
print(lambda_qut)
p2=20
num_rep = 1
learningRate_list=[0.01, 0.001]
iniscale=0.001




w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro,needles_index_hat,tpr,fdr,hat_equal_true,confusion_matrix,accuracy, ind_neurons, phat=computeResults(x_train_rescaled,y_train,learningRate_list,iniscale,lambda_qut,p2,x_test_rescaled,y_test) 

np.save("./w1o", w1o)

np.save("./phat"+numero, phat)
np.save("./ytestvrai_"+numero, y_test_hat_o )
np.save("./y_test_"+numero, y_test)
print("lambda : ", lro)
####################################################################################
time_end=time.time()
print("It takes " + str(time_end-time_start) + " seconds for "+str(num_rep)+" times." )

# w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro,needles_index_hat,tpr,fdr,hat_equal_true,confusion_matrix,accuracy=computeResults(x_train_rescaled,y_train,learningRate_list,iniscale,lambda_qut,p2,x_test_rescaled,y_test) 

print(needles_index_hat)

y_train=y_train.to_numpy()
#y_test_hat_o=y_test_hat_o.to_numpy()
number_class =y_train.shape[0]
###confusion :row-hat,colume-true
confusion_matrix_train= np.zeros([number_class,number_class])

number=np.sum(y_train,axis=1)
for i in np.arange(0,number_class):
    for j in np.arange(0,number_class):
       confusion_matrix_train[i,j]=np.sum(y_hat_o[i,np.where(y_train[j,]==1)])
accuracy_train=np.sum(np.diag(confusion_matrix_train))/n_train

print(accuracy_train, np.sum(np.diag(confusion_matrix)), n_train)
from sklearn.metrics import f1_score
#F1_score=f1_score(y_test, y_test_hat_o, average='macro')
dataset="fissures_handmade_test_max_pooling=2_4_types_d'images"
depth=2



y_test=y_test.to_numpy()
#y_test_hat_o=y_test_hat_o.to_numpy()
number_class =y_train.shape[0]
###confusion :row-hat,colume-true
confusion_matrix= np.zeros([number_class,number_class])

number=np.sum(y_test,axis=1)
for i in np.arange(0,number_class):
    for j in np.arange(0,number_class):
       confusion_matrix[i,j]=np.sum(y_test_hat_o[i,np.where(y_test[j,]==1)])

from sklearn.metrics import f1_score
#F1_score=f1_score(y_test, y_test_hat_o, average='macro')
dataset="MNIST_4_classes_"+numero


print(needles_index_hat)

with open("./results_ANN",'a') as file_con:
     file_con.write('dataset='+dataset+"\n")
     #file_con.write('depth='+ str(depth)+"\n")
     file_con.write('time='+ str(time_end-time_start)+"\n")
     file_con.write('lambda_qut='+str(lambda_qut)+"\n")
     file_con.write('cost_o='+str(cost_o)+"\n")
     file_con.write('accuracy_on_train='+str(accuracy_train)+"\n")
     file_con.write('accuracy_on_test='+str(accuracy)+"\n")
     file_con.write('confusion_matrix_train='+str(confusion_matrix_train)+"\n")
     file_con.write('confusion_matrix_test='+str(confusion_matrix)+"\n")
     #file_con.write('F1_score='+str(F1_score)+"\n")
     file_con.write("shape_needles="+str(np.shape(needles_index_hat)[1])+"\n")
     file_con.write("needles_index="+str(needles_index_hat)+"\n")
     file_con.write("neurons_index="+str(ind_neurons)+"\n")
     file_con.write("filters_index"+str( np.ceil(np.array(needles_index_hat)/float(p1)).astype(int)))
     
     file_con.write("\n")
     
