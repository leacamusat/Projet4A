import numpy as np 
import pandas as pd 
import sklearn 
from sklearn.utils import shuffle 
import statistics


	
#to load map of caracteristics which are the most activated 
# def extraction_filtre(nom_fichier_train, type=0):
# 	feature_maps=np.load(nom_fichier_train)
# 	print("there are ", feature_maps.shape[0], " train images for which there is ", feature_maps.shape[3], " features maps of size ", feature_maps.shape[1], "x", feature_maps.shape[2])
# 	if type==0.5:
# 		index_filter_kept=np.where(np.count_nonzero(feature_maps, axis=(0,1,2), keepdims=False)>statistics.median(np.count_nonzero(feature_maps, axis=(0,1,2), keepdims=False)))#keep only filter with activation map which have been activated by more than the median number of images which have usually activated the filter
# 	if type==0:
# 		index_filter_kept=np.where(np.count_nonzero(feature_maps, axis=(0,1,2), keepdims=False)!=0) #keep only filter only activation map which have at least been activated by one training image
		
# 	if type==0.75:
# 		index_filter_kept=np.where(np.count_nonzero(feature_maps, axis=(0,1,2), keepdims=False)>statistics.quantiles(np.count_nonzero(feature_maps, axis=(0,1,2), keepdims=False))[2])#keep only filter only activation map
# 	return feature_maps[:,:,:,index_filter_kept[0]]

#in the terminal sed 'y/,/ /' test.data > test_sans_virgule.data #permet d'enlever les virgules du code 

def creation_datasets(shuffl, rd_state, cut=False, labels_test="labels_test.csv", labels_train="labels_train.csv", activation_test="activations_test.txt.npy", activation_train="activations_train.txt.npy"):

	"""

        Generate 4 files adapted to be input either to DIMLP program or LASSO program : 
		1) file to feed DIMLP program of the activation maps flattened for test and train : 
		test.data, train.data
		2) file to feed DIMLP program of only activated activation maps flattened (non_zeros-map)
		 for test and train : non_zero_test.data, non_zero_train.data
		3) file to feed DIMLP program of only activation maps activated by at least  the half 
		of the training images for test and train : median_test.data, median_train.data
        4) file to feed DIMLP program of the activated activation maps flattened activated by at least
		 the 3rd quantile of the train-images for test and train : quantile_test.data, quantile_train.data

        Parameters
        ----------
                shuffl : boolean 
                	shuffle or not the lines of the matrix 
                cut : boolean 
					restriction of the number of individuals or not 
					(if True : keeps only the 4000 first individuals for the train 
					and the 2000 first individuals for the test, if False : keeps only the 2000 first)
                rd_state : random state which enables to keep the same selection of rows at each call-function
				label_test : string 
					name of the file of labels for test :  (0-1 labels with nb_class columns and nb_individuals rows)
				label_train : string 
					name of the file of labels for train :  (0-1 labels with nb_class columns and nb_individuals rows)
                activation_test : string 
					file with the activations map saved by the model for the test
				activation_train : 
					file with the activations map saved by the model for the train
        """

	labels=[labels_test, labels_train]
	final_datasets=["test.data", "train.data"]
	nb_classes=pd.read_csv(labels[0], header=None).to_numpy().shape[1]

	for i,j in enumerate([activation_test, activation_train]):
		feature_maps=np.load(j)
		feature_maps_train=np.load(activation_train)
		print("there are ", feature_maps.shape[0], " train images for which there is ", feature_maps.shape[3], " features maps of size ", feature_maps.shape[1], "x", feature_maps.shape[2])
		ind_kept=[]
		for type in [0.5, 0.75, 0]:
			if type==0.5:
				index_filter_kept=np.where(np.count_nonzero(feature_maps_train, axis=(0,1,2), keepdims=False)>statistics.median(np.count_nonzero(feature_maps_train, axis=(0,1,2), keepdims=False)))#keep only filter with activation map which have been activated by more than the median number of images which have usually activated the filter
				tot=np.concatenate((feature_maps[:,:,:,index_filter_kept[0]].reshape(feature_maps.shape[0],-1), pd.read_csv(labels[i], header=None).to_numpy()), axis=1)
				if shuffl:
					tot=shuffle(tot, random_state=rd_state)
				if cut: 
					if (i==1) :
						tot=tot[:4000, :] 
					else :
						tot=tot[:2000, :]
						ind_kept.append(index_filter_kept)										

				np.savetxt( "median_"+str(tot.shape[1]-nb_classes)+"_"+str(nb_classes)+final_datasets[i], tot, "%10.5f", delimiter=",")
			if type==0:
				index_filter_kept=np.where(np.count_nonzero(feature_maps_train, axis=(0,1,2), keepdims=False)!=0) #keep only filter only activation map which have at least been activated by one training image
				tot=np.concatenate((feature_maps[:,:,:,index_filter_kept[0]].reshape(feature_maps.shape[0],-1), pd.read_csv(labels[i], header=None).to_numpy()), axis=1)
				if shuffl:
					tot=shuffle(tot, random_state=rd_state)
				if cut : 
					if (i==1) :
						tot=tot[:4000, :] 
					else :
						tot=tot[:2000, :]
						ind_kept.append(index_filter_kept)										
				np.savetxt("non_zero_"+str(tot.shape[1]-nb_classes)+"_"+str(nb_classes)+final_datasets[i], tot, "%10.5f", delimiter=",")
				
			if type==0.75:
				index_filter_kept=np.where(np.count_nonzero(feature_maps_train, axis=(0,1,2), keepdims=False)>statistics.quantiles(np.count_nonzero(feature_maps_train, axis=(0,1,2), keepdims=False))[2])#keep only filter only activation map
				tot=np.concatenate((feature_maps[:,:,:,index_filter_kept[0]].reshape(feature_maps.shape[0],-1), pd.read_csv(labels[i], header=None).to_numpy()), axis=1)
				if shuffl:
					tot=shuffle(tot, random_state=rd_state)
				if cut : 
					if (i==1) :
						tot=tot[:4000, :] 
					else :
						tot=tot[:2000, :]
						ind_kept.append(index_filter_kept)										
				np.savetxt("quantile_"+str(tot.shape[1]-nb_classes)+"_"+str(nb_classes)+final_datasets[i], tot , "%10.5f", delimiter=",")
				
		tot=np.concatenate((feature_maps.reshape(feature_maps.shape[0],-1), pd.read_csv(labels[i], header=None).to_numpy()), axis=1)
		if shuffl:
			tot=shuffle(tot, random_state=rd_state)
		if cut : 
			if (i==1) :
				tot=tot[:4000, :] 
			else :
				tot=tot[:2000, :]
		np.savetxt(str(tot.shape[1])+final_datasets[i], tot, "%10.5f", delimiter=",")

creation_datasets(True, 50, cut=True)
