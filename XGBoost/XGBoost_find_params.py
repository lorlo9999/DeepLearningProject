import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# Specify the values to test, leave empty to be ignored
#======================================================
max_depth_range = [1,2,3,4,5,6,7,8]
# max_depth_range = 0
n_trees_range = np.arange(10,400,5)
# n_trees_range = 0
learning_rate_range = np.linspace(0.01,1)
# learning_rate_range = 0
reg_lambda_range = np.linspace(0,5,50)
#======================================================
# Figure settings
#======================================================
dpi = 400
show_figures = False

# Default Values

# Loading the Data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')

X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

if type(max_depth_range) != int:

    # 0: SF, 1: composite, 2: AGN, 3: LINER

    auc_list = [[], [], [], []]
    auc_avrg = []

    for max_depth in tqdm(max_depth_range):
        model = XGBClassifier(n_estimators = 40,
                            max_depth = max_depth,
                            objective = 'binary:logistic'
                            )

        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)

        for i in range(4):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            auc_list[i].append(roc_auc)

    auc_list = np.array(auc_list) 
    auc_avrg = np.mean(auc_list, axis=0)
    max_idx = np.argmax(auc_avrg)  # Get index of the max AUC value
    max_x = max_depth_range[max_idx]  # Corresponding x-axis value
    max_y = auc_avrg[max_idx]  # Maximum AUC value

    plt.figure()

    classes = ['Star-Forming', 'Composite', 'AGN', 'LINER']

    for i in range(4):
        plt.plot(max_depth_range,auc_list[i], label=f'{classes[i]}')
    
    plt.plot(max_depth_range,auc_avrg, label='Average AUC', linestyle='--') # The Average

    plt.scatter(max_x, max_y, color='red', zorder=3, label=f'Max AUC ({max_y:.3f})')  # Highlight point
    plt.axvline(x=max_x, linestyle=':')
    plt.text(max_x + 0.005, max_y - 0.01, f'Best tree depth: {max_x}', ha='left', fontsize=10)

    plt.xlabel('Maximum depth per tree')
    plt.ylabel('AUC score')
    plt.title('Variation of the Tree Depth')

    plt.legend(loc='best')
    plt.savefig('../plots/XGB_best_tree_depth.pdf', dpi=dpi)
    if show_figures: plt.show()
    best_max_depth = max_x


if type(n_trees_range) != int:

    # 0: SF, 1: composite, 2: AGN, 3: LINER

    auc_list = [[], [], [], []]
    auc_avrg = []

    for n_trees in tqdm(n_trees_range):
        model = XGBClassifier(n_estimators = int(n_trees),
                            max_depth = best_max_depth,
                            objective = 'binary:logistic',
                            )

        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)

        for i in range(4):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            auc_list[i].append(roc_auc)

    auc_list = np.array(auc_list)  
    auc_avrg = np.mean(auc_list, axis=0)
    max_idx = np.argmax(auc_avrg)  # Get index of the max AUC value
    max_x = n_trees_range[max_idx]  # Corresponding x-axis value
    max_y = auc_avrg[max_idx]  # Maximum AUC value

    plt.figure()

    classes = ['Star-Forming', 'Composite', 'AGN', 'LINER']

    for i in range(4):
        plt.plot(n_trees_range,auc_list[i], label=f'{classes[i]}')
    
    plt.plot(n_trees_range,auc_avrg, label='Average AUC', linestyle='--') # The Average

    plt.scatter(max_x, max_y, color='red', zorder=3, label=f'Max AUC ({max_y:.3f})')  # Highlight point
    plt.axvline(x=max_x, linestyle=':')
    plt.text(max_x + 0.005, max_y - 0.01, f'Best amount: {max_x}', ha='left', fontsize=10)

    plt.xlabel('Number of Subtrees')
    plt.ylabel('AUC score')
    plt.title('Variation of the Amount of Subtrees')

    plt.legend(loc='best')
    plt.savefig('../plots/XGB_best_n_trees.pdf', dpi=dpi)
    if show_figures: plt.show()

    best_n_trees = max_x

if type(learning_rate_range) != int:

    # 0: SF, 1: composite, 2: AGN, 3: LINER

    auc_list = [[], [], [], []]
    auc_avrg = []

    for learning_rate in tqdm(learning_rate_range):
        model = XGBClassifier(n_estimators = best_n_trees,
                            max_depth = best_max_depth,
                            objective = 'binary:logistic',
                            learning_rate = learning_rate
                            )

        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)

        for i in range(4):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            auc_list[i].append(roc_auc)

    auc_list = np.array(auc_list) 
    auc_avrg = np.mean(auc_list, axis=0)
    max_idx = np.argmax(auc_avrg)  # Get index of the max AUC value
    max_x = learning_rate_range[max_idx]  # Corresponding x-axis value
    max_y = auc_avrg[max_idx]  # Maximum AUC value

    plt.figure()

    classes = ['Star-Forming', 'Composite', 'AGN', 'LINER']

    for i in range(4):
        plt.plot(learning_rate_range,auc_list[i], label=f'{classes[i]}')
    
    plt.plot(learning_rate_range,auc_avrg, label='Average AUC', linestyle='--') # The Average

    plt.scatter(max_x, max_y, color='red', zorder=3, label=f'Max AUC ({max_y:.3f})')  # Highlight point
    plt.axvline(x=max_x, linestyle=':')
    plt.text(max_x + 0.005, max_y - 0.014, f'Best learning rate: {max_x:.2f}', ha='left', fontsize=10)

    plt.xlabel('Learning rate')
    plt.ylabel('AUC score')
    plt.title('Variation of the Learning Rate')

    plt.legend(loc='lower right')
    plt.savefig('../plots/XGB_best_learning_rate.pdf', dpi=dpi)
    if show_figures: plt.show()

    best_learning_rate = max_x

if type(reg_lambda_range) != int:

    # 0: SF, 1: composite, 2: AGN, 3: LINER

    auc_list = [[], [], [], []]
    auc_avrg = []

    for reg_lambda in tqdm(reg_lambda_range):
        model = XGBClassifier(n_estimators = best_n_trees,
                            max_depth = best_max_depth,
                            objective = 'binary:logistic',
                            learning_rate = best_learning_rate,
                            reg_lambda = reg_lambda
                            )

        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)

        for i in range(4):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            auc_list[i].append(roc_auc)

    auc_list = np.array(auc_list) 
    auc_avrg = np.mean(auc_list, axis=0)
    max_idx = np.argmax(auc_avrg)  # Get index of the max AUC value
    max_x = reg_lambda_range[max_idx]  # Corresponding x-axis value
    max_y = auc_avrg[max_idx]  # Maximum AUC value

    plt.figure()

    classes = ['Star-Forming', 'Composite', 'AGN', 'LINER']

    for i in range(4):
        plt.plot(reg_lambda_range,auc_list[i], label=f'{classes[i]}')
    
    plt.plot(reg_lambda_range,auc_avrg, label='Average AUC', linestyle='--') # The Average

    plt.scatter(max_x, max_y, color='red', zorder=3, label=f'Max AUC ({max_y:.3f})')  # Highlight point
    plt.axvline(x=max_x, linestyle=':')
    plt.text(max_x - 0.005, max_y - 0.008, r'Best $\lambda$ : {:.2f}'.format(max_x), ha='right', fontsize=10)

    plt.xlabel(r'$\lambda')
    plt.ylabel('AUC score')
    plt.title(r'Variation of the Regularization Parameter $\lambda$')

    plt.legend(loc='best')
    plt.savefig('../plots/XGB_best_reg_lambda.pdf', dpi=dpi)
    if show_figures: plt.show()