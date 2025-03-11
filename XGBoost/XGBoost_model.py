import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#=========================================
# Parameters to Change || Default Value
#=========================================
max_depth = 3 # Maximum allowed  Tree depth. To high values lead to overfitting || 6
n_trees = 30 # Number of trees created || 100
learning_rate = 0.21 # Eta || 0.3 (Optimal should be 0.11, but testing showed otherwise)
reg_lambda = 1.63 # L2 regularization term || 1 
#=========================================

# Loading the Data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')

X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

model = XGBClassifier(n_estimators = n_trees,
                     max_depth = max_depth,
                     objective = 'binary:logistic',
                     learning_rate = learning_rate,
                     reg_lambda = reg_lambda,
                     )

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()

classes = ['Star-Forming', 'Composite', 'AGN', 'LINER']

for i in range(4):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()