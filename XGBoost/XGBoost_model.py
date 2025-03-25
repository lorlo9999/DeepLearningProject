import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

# Training
model.fit(X_train, y_train)

model.save_model('../XGBoost/XGB_saved_model.json')

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

classes = ['SF', 'Composite', 'AGN', 'LINER']

# === PLOT 1 ===
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
plt.title('ROC Curve')
plt.tight_layout()
plt.legend(loc='lower right')
plt.savefig('../plots/XGB_ROC.pdf', dpi=400, bbox_inches='tight')
plt.savefig('../plots/XGB_ROC.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

# === PLOT 2 ===
# Plotting the Feature importance for the XGBC 

feature = ['O3_index', 'O2_index', 'sigma_star', 'sigma_o3', 'u_g', 'g_r', 'r_i', 'i_z']
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
ordered_feature_names = []
ordered_feature_values = []
for i in indices:
    ordered_feature_names.append(feature[i])
    ordered_feature_values.append(importances[i])

print(ordered_feature_names)

plt.figure()

plt.barh(ordered_feature_names, ordered_feature_values, color='green')
plt.tight_layout(pad=2)
plt.xlabel('Feature Value')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.savefig('../plots/XGB_feat_importance.pdf', dpi=400, bbox_inches='tight')
plt.savefig('../plots/XGB_feat_importance.png', dpi=400, bbox_inches='tight')
plt.show()

# === PLOT 3 ===
# Visualizing the most important features

top2_idx = np.argsort(importances)[-2:]
X_top2 = X_train[:, top2_idx]

# I want to color in the classes in the plot
y_classes = np.argmax(y_train, axis=1)  # Convert to class labels

# Scatter plot
plt.figure()
cmap = plt.cm.tab10
for i, c in enumerate(classes):
    idx = np.where(y_classes == i)
    plt.scatter(x=X_top2[idx, 0], y=X_top2[idx, 1], color=cmap(i), label=c, alpha=0.7, s=4)

# Labels and title
plt.xlabel(r'$\log\sigma*$')
plt.ylabel(r'$\log [OII]/H\beta$')
plt.title('Distribution of the two Most Important Features')
plt.legend(loc='best',title='Actual Classes')
plt.tight_layout()
plt.savefig('../plots/XGB_top2_features.pdf', dpi=400)
plt.savefig('../plots/XGB_top2_features.png', dpi=400)
plt.show()

# === PLOT 4 ===
# Classification Report

y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1) 
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure()
sns.heatmap(cm, annot=True, fmt='.2f', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
plt.title('XGBClassifier Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('../plots/XGB_confusion_matrix.pdf', dpi=400)
plt.savefig('../plots/XGB_confusion_matrix.png', dpi=400)
plt.show()


print(classification_report(y_true, y_pred, target_names=classes))
per_class_accuracy = cm.diagonal()/cm.sum(axis=1)
print('Accuracy per class:')
for i,acc in enumerate(per_class_accuracy):
    print(f'{classes[i]}: {acc:.3f}')



