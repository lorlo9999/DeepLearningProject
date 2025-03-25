import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.utils import resample
from scipy.interpolate import interp1d

# Loading the Data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

# Initialize and train KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, np.argmax(y_train, axis=1))

y_pred_proba = knn.predict_proba(X_test)

test_labels = np.argmax(y_test, axis=1)
n_bootstraps = 100
rng = np.random.RandomState(42)

all_fprs = {i: [] for i in range(4)}
all_tprs = {i: [] for i in range(4)}
mean_fpr_interpolated = np.linspace(0, 1, 100)

for i in range(4):
    for _ in range(n_bootstraps):
        X_resampled, y_resampled = resample(X_test, test_labels, replace=True, random_state=rng)
        y_prob_resampled = knn.predict_proba(X_resampled)
        fpr_resampled, tpr_resampled, _ = roc_curve((y_resampled == i).astype(int), y_prob_resampled[:, i])
        f_interp = interp1d(fpr_resampled, tpr_resampled, kind='linear', fill_value="extrapolate")
        tpr_interpolated = f_interp(mean_fpr_interpolated)
        all_fprs[i].append(mean_fpr_interpolated)
        all_tprs[i].append(tpr_interpolated)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve((test_labels == i).astype(int), y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
classes = ['Star-Forming', 'Composite', 'AGN', 'LINER']

for i in range(4):
    all_fprs[i] = np.array(all_fprs[i])
    all_tprs[i] = np.array(all_tprs[i])
    mean_tpr = np.mean(all_tprs[i], axis=0)
    mean_fpr = mean_fpr_interpolated
    std_tpr = np.std(all_tprs[i], axis=0)
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.5)
    plt.plot(mean_fpr, mean_tpr, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})', lw=2)

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.tight_layout(pad=2)
plt.ylabel('True Positive Rate')
plt.title('k-Nearest Neighbour ROC')
plt.legend(loc='lower right')
plt.savefig('../plots/KNN_ROC.pdf', dpi=400)
plt.show()

# Classification Report
y_pred = knn.predict(X_test)
print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred, target_names=classes))
