import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from tensorflow.keras.utils import to_categorical

n_input_pars=8
n_output_pars=4
file_path_features = 'data/X_extra_data.npy'
X_test = np.load(file_path_features)
file_path_labels = 'data/labels_extra_data.npy'
Y_test = np.load(file_path_labels)
Y_test = np.array(Y_test, dtype=int)
model_name='model_rf_5.sav'
n_source = X_test.shape[0]
class_names = ['SF', 'COMP', 'AGN', 'LINER']

# load the model from disk
model = pickle.load(open(model_name, 'rb'))

type_arr_out0=np.array(model.predict(X_test))
model=0

type_arr_out=[]
for i in range(n_source):
    type_this=np.array([float(type_arr_out0[i])])
    counts = np.bincount(type_this.astype(int))
    type_arr_out.append(np.argmax(counts))
type_arr_out=np.array(type_arr_out)

# ind_new_sf=np.where(type_arr_out ==1)
# ind_new_comp=np.where(type_arr_out ==2)
# ind_new_AGN=np.where(type_arr_out ==3)
# ind_new_liner=np.where(type_arr_out ==4)
# cm = confusion_matrix(Y_test, type_arr_out)
type_arr_out = to_categorical(type_arr_out-1, num_classes=4)

# # Plot the roc curve
from sklearn.metrics import roc_curve, auc
fpr = {}
tpr = {}
roc_auc = {}
tpr_uncertainty = {}

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(Y_test[i], type_arr_out[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))

for i in range(4):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    # plt.fill_between(
    #     fpr[i],
    #     np.maximum(tpr[i] - tpr_uncertainty[i], 0),
    #     np.minimum(tpr[i] + tpr_uncertainty[i], 1),
    #     alpha=0.2
    # )

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Uncertainty')
plt.legend(loc='lower right')
plt.savefig('Classifications/roc_curve_classifier.pdf')
plt.show()



