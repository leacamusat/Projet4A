#!/bin/sh

#SBATCH --cpus-per-task=1
#SBATCH --job-name=testTensorFlow
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%J.out
#SBATCH --gpus=1
#SBATCH --partition=shared-gpu
#SBATCH --extra-node-info=1:32:1
#SBATCH --mem=12000
## TensorFlow
module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0

## CUDA
module load cuDNN/8.2.1.32-CUDA-11.3.1

rm -r project 

virtualenv project

source project/bin/activate
#installe tous les packages nécessaires à l'exécution, qui sont listés dans le fichier requirement 
pip install -U pip
pip uninstall scikit-learn
pip install scikit-learn==1.1.1
pip uninstall pytz
pip install pytz==2022.1
pip uninstall six 
pip install six==1.16.0
pip install matplotlib==3.5.2
pip install opencv-python==4.6.0.66
pip install opencv-contrib-python==4.6.0.66
pip install -v -r  ./requirements.txt
pip install split-folders
#run le fichier 

#srun python ./model_extraction_caracteristic_feature.py 2 1 2

#srun python ./generation_image
#srun ./DimlpTrn -L ../GPU/train_convol.data -T ../GPU/test_convol.data -I 40000 -O 2 -R 
#srun python ./generation_image
#srun python ./model_FISSURES_extraction_caracteristic_feature.py 1 2 2
srun python ./main_ann_lasso_classification_2layer.py ./MNIST_4 35 4
###################### ATTENTION : pour appeler ce .sh dans un terminal avec python faire un sbatch baobab.sh pour ne pas exécuter sur le noeud principal !!! #######################################
