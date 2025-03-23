import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix

#=========================================
# Parameters to Change || Default Value
#=========================================
# max_depth = 3 # Maximum allowed  Tree depth. To high values lead to overfitting || 6
# n_trees = 30 # Number of trees created || 100
# learning_rate = 0.21 # Eta || 0.3 (Optimal should be 0.11, but testing showed otherwise)
# reg_lambda = 1.63 # L2 regularization term || 1 
#=========================================


# Loading the Data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')

X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

model = XGBClassifier(objective = 'binary:logistic')

# Training
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)

# Prediction with uncertainty

n_bootstraps = 100
rng = np.random.RandomState(42)


all_fprs = {i: [] for i in range(4)}
all_tprs = {i: [] for i in range(4)}

mean_fpr_interpolated = np.linspace(0, 1, 100)

for i in range(4):
    for _ in range(n_bootstraps):

        X_resampled, y_resampled = resample(X_test, y_test, replace=True, random_state=rng)
        
        y_prob_resampled = model.predict_proba(X_resampled)
        
        fpr_resampled, tpr_resampled, _ = roc_curve(y_resampled[:, i], y_prob_resampled[:, i])
        
        f_interp = interp1d(fpr_resampled, tpr_resampled, kind='linear', fill_value="extrapolate")
        tpr_interpolated = f_interp(mean_fpr_interpolated)
        
        all_fprs[i].append(mean_fpr_interpolated)
        all_tprs[i].append(tpr_interpolated)


fpr = {}
tpr = {}
roc_auc = {}

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()

classes = ['Star-Forming', 'Composite', 'AGN', 'LINER']

# Plotting the ROC curve with uncertainties
for i in range(4):
    
    all_fprs[i] = np.array(all_fprs[i])
    all_tprs[i] = np.array(all_tprs[i])
    
    
    mean_tpr = np.mean(all_tprs[i], axis=0)
    mean_fpr = mean_fpr_interpolated  

    
    std_tpr = np.std(all_tprs[i], axis=0)
    nan_to_zero = np.where(np.isnan(std_tpr), 0, std_tpr)
    std_tpr = nan_to_zero

    
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,alpha=0.5)
    
    
    plt.plot(mean_fpr, mean_tpr, label=rf'{classes[i]} (AUC = {roc_auc[i]:.3f}; $\sigma$ = {np.mean(std_tpr):.3f})', lw=2)

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGB ROC Curve with Default Settings')

plt.legend(loc='lower right')
plt.savefig('../plots/XGB_default.pdf', dpi=400)
plt.show()
plt.close()

# Classification Report
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1) 
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=classes))