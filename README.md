# Computational Astrophysics

### This repository contains code and data for classifying spectra from the Digitized First Byurakan Survey (DFBS).


### Classification

This folder contains code and data for training and testing convolutional neural network models for classifying spectra into four classes: UV-excess galaxies, hot subdwarfs, carbon stars, and other objects. The code is the implementation of the paper: https://doi.org/10.1016/j.ascom.2020.100442.


### Sub-Object Classification

This folder contains code and data for training and testing convolutional neural network models for the sub-object classification of the abovementioned groups. 


### Cloud Service

This folder contains code and data for the cloud-based service for classifying astronomical images in a Google Colab environment. The service allows users to upload their spectra or use sample spectra from DFBS and get the classification results. The folder has three main notebooks: train_colab.ipynb, test_colab.ipynb, and infer_colab.ipynb.
