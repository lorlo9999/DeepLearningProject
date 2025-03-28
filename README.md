# Improving the Classification for Low Redshift Emission-line Galaxies

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
2. composite galaxy
3. active galactic nuclei (AGN)
4. low-ionization nuclear emission region (LINER)


In this project we are using different approaches to improve the accuracy of the red-shift source predictions. The following methods were used:
- using a more sophisticated neural network
- improving the random-forest model
- using gradient boosted decision tree learning (XGBoost)

We managed to achieve improvements with all approaches. 

We also recreated the results for each method that the paper used. Additionally, we tried classifying intermediate redshift galaxies that we obtained from the SDSS, however, the modelds did nto generalise well onto different data.

<span style="color:red">**Part about applying the models on the new data**</span>.

## The Repository

| Filename | Description |
|-----------|-------------|
| *new_classifier.py* | Python file classifying the extra intermediate redshift data. |
| *plotting.py* | Python file containing functions to simplify the plotting. |
| *roc_helper.py* | Python functions for easy ROC plotting. |
| *sckikit_kfold_classifier.py* | The original code for using a trained model to make classifications to 0.32<z<0.8 emission line galaxies. |
| *scikit_kfold_training* | The original code for training the models, with our comments added to make it more clear. |
| *Slides_DLiP.pdf* | PDF file of the slides from our presentation. |
| *sort_data.py* | Python file to create the input features and labels from `data_matched_step2_newz_sm.csv`. It produces `sizes,npy`, `X_test.npy`, `X_train.npy`, `y_test.npy` and `y_train.npy`. |
| *sort_extra_data.ipynb* | Python notebook to sort the extra intermediate redshift (classified) data we obtained from SDSS. It produces `X_extra_data.npy` and `labels_extra_data.npy`. |

### > data
| Filename | Description |
|-----------|-------------|
| *data_elg.csv* | CSV containing input parameters for test sample used by scikit_kfold_classifier.py . |
| *data_matched_step2_newz_sm.csv* | File containing the input parameters for training. More info in its header. |
| *eboss-elg-classification.fits* | File containing 0.32<z<0.8 emission line galaxies, no classification. |
| *extra_data_50000_z0.3-0.8_classified.csv* | The extra 0.32<z<0.8 and already classified data from the SDSS database. |
| *labels_extra_data.npy* | Numpy array with the labels of the additional data. This file and the following data are saved as a numpy arrays so they can directly be used without any more data manipulation.|
| *sizes.npy* | Numpy array containing the distribution of the output classes in the training data. |
| *X_extra_data.npy* | Numpy array with corresponding input values. |
| *X_test.npy* | Numpy array with test input data for z<0.3 <span style="color:red">correct?</span>.|
| *X_train.npy* | Numpy array with the training input data.|
| *y_test.npy* | Numpy array with the correct labels for `X_test.npy`. |
| *y_train.npy* | Numpy array with the correct labels for `X_train.npy`. |

### > KNN
| Filename | Description |
|-----------|-------------|
| *knn_model.py* | A simple recreation of their knn model. |

### > NeuralNetwork
| Filename | Description |
|-----------|-------------|
| *NeuralNetwork.ipynb* | Jupyter notebook for the training and testing of the neural network. |
| *NN_model.keras* | The saved complete neural network model for easy importation and using. |

| *ColorsNN.ipynb* | Jupyter notebook for training and testing of neural network limited to the 4 color features. |
| *OpticalNN.ipynb* | Jupyter notebook for training and testing of neural network limited to the 4 optical features. |

### > plots

This folder contains all of the plots as PDF files. 

### > RandomForest

| Filename | Description |
|-----------|-------------|
| *RandomForest.ipynb* | Jupyter notebook for the training and testing of the random forest model. |
| *random_forest_model.pkl* | The saved RF model, currently not included as it was too heavy to push. If the notebook is run it will appear here. |

### > SVC
| Filename | Description |
|-----------|-------------|
| *SVC_model.py* | A simple recreation of their SVC model. |

### > XGBoost

| Filename | Description |
|-----------|-------------|
| *XGB_default.py* | Python file to run the XGBClassifier with its default settings. |
| *XGB_saved_model.json* | File containing the fitted XGBoost classifier. |
| *XGBoost_find_params.py* | Python file to determine the optimal hyperparameters for creating the XGBClassifier. |
| *XGBoosted_model.py* | Python file for creating and evaluating the XGBClassifier. |
| *XGBoost_new_data.py* | Python file to load the XGBClassifier and predict the intermediate redshift sources with it.|

