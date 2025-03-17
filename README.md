# Improving the Classification for Intermediate Redshift Emission-line Galaxies

Repository for the final project in "Deep Learning in Physics".
This project is based on improving the results of [Zhang et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...883...63Z/abstract), the corresponding GitHub repository can be found [here](https://github.com/zkdtc/MLC_ELGs)


## Contributors:

**Name - Student ID - GitHub Username**

Lorenzo Cesario - S4220110 - lorlo9999

Csilla Tijssen - S4103246 - Csilla42

Felix Schulz - S6178979 - Fubinii

## Overview

For an in-depth description of the classification goal, please refer to the original paper. In a nutshell: we are using machine learning to predict the source of measured redshift emission lines. Our input features are `[O III]/Hβ`, `[O II]/Hβ`, linewidth `σ([O III])`; stellar velocity dispersion `σ*` and four colors `u − g`, `g − r`, `r − i`, and `i − z`. 
The classification labels are
1. star-forming galaxy (SFG)
2. somposite galaxy
3. active galactic nuclei (AGN)
4. low-ionization nuclear emission region (LINER)


In this project we are using different approaches to improve the accuracy of the red-shift source predictions. Namely the following methods were used:
- using a more sophisticated neural network
- improving the random-forest model
- using gradient boosted decision tree learning (XGBoost)

We managed to achieve improvements with all approaches. 

<span style="color:red">**Part about applying the models on the new data**</span>.

## The Repository

| Filename | Description |
|-----------|-------------|
| *plotting.py* | Python file containing functions to simplify the plotting. |
| *roc_helper.py* | Python functions for easy ROC plotting. |
| *sckikit_kfold_classifier.py* | The original code for using a trained model to make classifications to 0.32<z<0.8 emission line galaxies. |
| *scikit_kfold_training_with_comments* | <span style="color:red">Csilla please write something here</span>. |
| *sort_data.py* | Python file to create the input features and labels from `data_matched_step2_newz_sm.csv`. It produces `sizes,npy`, `X_test.npy`, `X_train.npy`, `y_test.npy` and `y_train.npy`. |

### > data
| Filename | Description |
|-----------|-------------|
| *data_elg.csv* | CSV containing input parameters for test sample used by scikit_kfold_classifier.py . |
| *data_matched_step2_newz_sm.csv* | File containing the input parameters for training. More info in its header. |
| *eboss-elg-classification.fits* | File containing 0.32<z<0.8 emission line galaxies, no classification. |
| *extra_data_50000_z0.3-0.8_classified.csv* | <span style="color:red">Missing</span>|
| *labels_extra_data.npy* | Numpy array with the labels of the additional data. This file and the following data are saved as a numpy arrays so they can directly be used without any more data manipulation.|
| *sizes.npy* | Numpy array containing the distribution of the output classes in the training data. |
| *X_extra_data.npy* | Numpy array with corresponding input values. |
| *X_test.npy* | Numpy array with test input data for z<0.3 <span style="color:red">correct?</span>.|
| *X_train.npy* | Numpy array with the training input data.|
| *y_test.npy* | Numpy array with the correct labels for `X_test.npy`. |
| *y_train.npy* | Numpy array with the correct labels for `X_train.npy`. |

### > NeuralNetwork
| Filename | Description |
|-----------|-------------|
| *NeuralNetwork.ipynb* | Jupyter notebook for the training and evaluation of the neural network. |
| *NN.h5* | The weights of the trained network.|

### > plots

This folder contains all of the plots as PDF files. 

### > RandomForest

| Filename | Description |
|-----------|-------------|
| *RandomForest.ipynb* | Jupyter notebook for the training and evaluation of the random forest model. |

### > XGBoost

| Filename | Description |
|-----------|-------------|
| *XGB_saved_model.json* | File containing the fitted XGBoost classifier. |
| *XGBoost_find_params.py* | Python file to determine the optimal hyperparameters for creating the XGBClassifier. |
| *XGBoosted_model.py* | Python file for creating and evaluating the XGBClassifier. |

