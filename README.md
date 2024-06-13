# ECE228Project

The notebook to run the experiments on Synthetic Data is in SyntheticExperiment.ipynb and for the Melanoma dataset is in Melanoma.ipynb (remember to update the data_dir to your data directory) . The Ours BBSE refers to our actual method (Ours is the performance of our method before test adaptation). 

scripts folder has the implementation of all baselines. To run the SyntheticExperiment.ipynb, comment out lines 30-42 (this will ensure the appropriate neural network for this data). To run melanoma, nothing has to be changed in this folder.

To get access to the melanoma dataset, download the data from - https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256.
Run the melanoma_resnet_features.py in the data folder (update the data_dir to your data directory). This will save a numpy file called resnet_features.npy in your folder.

For any questions, please email pprashant@ucsd.edu
