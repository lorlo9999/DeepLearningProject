import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Loading the trained XGB model
model = XGBClassifier()
model.load_model('XGB_saved_model.json')

# Preparing the new data for 0.3 < z < 0.8
X_xdata = np.load('../data/X_extra_data.npy')
y_xdata = np.load('../data/labels_extra_data.npy')

y_pred = model.predict_proba(X_xdata) # The new predictions

# Plotting the ROC curve
fpr = {}
tpr = {}
roc_auc = {}

plt.figure()

classes = ['SF', 'Composite', 'AGN', 'LINER']

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_xdata[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for the Additional Data')

plt.legend(loc='lower right')
plt.savefig('../plots/XGB_extra_data.pdf', dpi=400)
plt.savefig('../plots/XGB_extra_data.png', dpi=400)
plt.show()
plt.close()



