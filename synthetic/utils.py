# Contains all the function for the synthetic experiements

# Import all libraries
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale = 2, rc={"figure.dpi":700, 'savefig.dpi':300})

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


### Display functions
def display_data(data, labels, protected, colors = ['tab:orange', 'tab:blue'], legend = True, distribution = False):
    """
        Displays data distribution.

        Args:
            data (Array n * 2): Two dimensional frame of the coordinates of each points.
            labels (Array n * 1): Label associated to each point.
            protected (Array n * 1): Group membership (Contains 'Minority' or 'Majority').
            colors (list, optional): Colors for each group. Defaults to ['tab:orange', 'tab:blue'].
            legend (bool, optional): Displays legend. Defaults to True.
            distribution (bool, optional): Computes kde if true. Defaults to False.
    """
    # Iterate through group and display data
    for name, color in zip(['Majority', 'Minority'], colors):
        data_p, label_p = data[protected == name], labels[protected == name]

        if distribution:
            sns.kdeplot(x = data_p.iloc[:, 0][~label_p], y = data_p.iloc[:, 1][~label_p], color = color, alpha = 0.7, linestyles = 'dashed', linewidth = 2, levels = 5)
            sns.kdeplot(x = data_p.iloc[:, 0][label_p], y = data_p.iloc[:, 1][label_p], color = color, alpha = 0.7, linestyles = 'dotted', linewidth = 2, levels = 5)
        else:
            plt.scatter(data_p.iloc[:, 0][~label_p], data_p.iloc[:, 1][~label_p], alpha = 0.25, label = name, c = color)
            plt.scatter(data_p.iloc[:, 0][label_p], data_p.iloc[:, 1][label_p], alpha = 0.25, marker = 'x', c = color)

    # Formatting
    plt.scatter([],[], label = 'Group', alpha = 0)
    plt.scatter([],[], label = 'Minority', c = 'tab:blue', alpha = 0.7)
    plt.scatter([],[], label = 'Majority', c = 'tab:orange', alpha = 0.7)
    plt.scatter([],[], label = ' ', alpha = 0)
    plt.scatter([],[], label = 'Class', alpha = 0)
    plt.plot([],[], ls = 'dotted', label = 'Positive', c = 'k')
    plt.plot([],[], ls = 'dashed', label = 'Negative', c = 'k')
    plt.xlabel(r'$x_2$')
    plt.ylabel(r'$x_1$')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)

    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def display_result(performance, type = 'AUC', legend = True, colors = ['tab:orange', 'tab:blue', 'tab:gray'], alphas = None):
    """
        Computes and displays model's performance and normal confidence bounds.

        Args:
            performance (Dict: str: pd.DataFrame): Dictionary of metrics (key is method, value is frame 
                of prediction with columns being the group of interest and line metrics over the experiments
                This dictionary results from k_experiment function)
            type (str, optional): Performance to be displayed. Defaults to 'AUC'.
            legend (bool, optional): Display legend (option for paper). Defaults to True.
            colors (list of matplotlib colors, optional): Colors for each groups. Defaults to ['tab:orange', 'tab:blue', 'tab:gray'].
            alphas (list of float, optional): Alphas to use for each methods. Defaults to None.
    """
    assert (alphas is None) or (len(performance) == len(alphas)), 'Not enough transparency provided (alphas = None for non transparency)'

    # Compute average
    mean, ci = {}, {}
    for method in performance:
        mean[method], ci[method] = {}, {}
        for group in performance[method].columns:
            assert len(performance[method].columns) == len(colors), 'Not enough colors provided'
            meth_group = performance[method][group]
            meth_group = meth_group[meth_group.index.get_level_values('Metric') == type]
            mean[method][group] = meth_group.mean()
            ci[method][group] = 1.96 * meth_group.std() / np.sqrt(len(meth_group))

    mean, ci = pd.DataFrame.from_dict(mean), pd.DataFrame.from_dict(ci)
    print(pd.DataFrame.from_dict({m: ["{:.3f} ({:.3f})".format(mean.loc[m].loc[i], ci.loc[m].loc[i]) for i in mean.columns] for m in mean.index}, columns = mean.columns, orient = 'index').to_latex())
    ax = mean.plot.barh(xerr = ci, legend = legend, figsize = (7, 7))
    # Change colors
    hatches = ['', 'ooo', 'xx', '//', '||', '***', '++']
    for i, thisbar in enumerate(ax.patches):
        c = list(plt_colors.to_rgba(colors[i % len(mean)]))
        c[3] = 1 if alphas is None else alphas[i // len(mean)]
        thisbar.set(edgecolor = '#eaeaf2', facecolor = c, linewidth = 1, hatch = hatches[i // len(mean)])
        
    plt.grid(alpha = 0.3)
    
    if type == 'AUC':
        plt.xlim(0.2, 1.0)
        plt.axvline(0.5, ls = ':', c = 'k', alpha = 0.5)
    else:
        plt.xlim(0., 0.8)
    plt.xlabel(type)

    if legend:
        ncol = len(np.unique(colors))
        patches = [ax.patches[i * len(mean) + j] for j in range(ncol) for i in range(len(mean.columns))][::-1]
        labels = [''] * (len(patches) - len(mean.columns)) + mean.columns.tolist()[::-1]
        ax.legend(patches, labels, loc='center left', bbox_to_anchor=(1, 0.5),
            title = 'Imputation strategies', ncol = ncol, handletextpad = 0.5, handlelength = 1.0, columnspacing = -0.5,)


### Generating functions
def generate_data_linear_shift(majority_size, ratio, class_balance = 0.5, seed = 0):
    """
        Generate data with a shift in the linear separation for the minority
        Negative are the same but not positive

        Args:
            majority_size (int): Number points for majority
            ratio (float): Ratio for minority (0.1 = 1 in minority for 10 in majority)
            class_balance (float, optional): Balance between class. Default 0.5.
    """
    np.random.seed(seed)

    majority_pos = 0.25 * np.random.randn(int(majority_size * class_balance), 2)
    majority_neg = 0.25 * np.random.randn(int(majority_size * (1 - class_balance)), 2)
    majority_pos[:, 0] += 1
    majority = np.concatenate([majority_pos, majority_neg])
    labels_maj = np.concatenate([np.full(len(majority_pos), True), np.full(len(majority_neg), False)])

    minority_pos = 0.25 * np.random.randn(int(majority_size * ratio * class_balance), 2)
    minority_neg = 0.25 * np.random.randn(int(majority_size * ratio * (1 - class_balance)), 2)
    minority_pos[:, 1] += 1
    minority = np.concatenate([minority_pos, minority_neg])
    labels_min = np.concatenate([np.full(len(minority_pos), True), np.full(len(minority_neg), False)])

    concatenation = np.concatenate([majority, minority])
    labels = np.concatenate([labels_maj, labels_min])
    protected =  np.concatenate([np.full(len(labels_maj), False), np.full(len(labels_min), True)])

    sort = np.arange(len(concatenation))
    np.random.shuffle(sort)
    return pd.DataFrame(concatenation[sort]), pd.Series(labels[sort]), pd.Series(protected[sort]), pd.Series(protected[sort]).replace({True: 'Minority', False: 'Majority'})



### Modelling function
def evaluate(y_true, groups, y_pred):
    """
        Computes the different metrics of interest.

        Args:
            y_true (Array n * 2): Array of ground truth labels.
            groups (Array n): Array of group membership.
            y_pred (Array n * 2): Array of predicted outcomes.

        Returns:
            Frame: Dataframe with brier score, auc and fixed FPR and TPR.
    """
    groups_unique = np.unique(groups).tolist() + ["Overall"]

    # Overall ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    fpr_sort = np.argsort(fpr)
    tpr_sort = np.argsort(tpr)
    threshold_fpr = np.interp(0.9, tpr[tpr_sort], thresholds[tpr_sort])
    threshold_tpr = np.interp(0.1, fpr[fpr_sort], thresholds[fpr_sort])

    performance = {}
    for group in groups_unique:
        if group == 'Overall':
            y_pred_group = y_pred
            y_true_group = y_true
        else:
            y_pred_group = y_pred[groups == group]
            y_true_group = y_true[groups == group]

        fpr, tpr, thresholds = roc_curve(y_true_group, y_pred_group)
        thres_order = np.argsort(thresholds)

        performance[group] = pd.Series({
            "Brier Score": brier_score_loss(y_true_group, y_pred_group),
            "AUC": roc_auc_score(y_true_group, y_pred_group),
            
            "FPR @ 90% TPR": np.interp(threshold_fpr, thresholds[thres_order], fpr[thres_order]),
            "TPR @ 10% FPR": np.interp(threshold_tpr, thresholds[thres_order], tpr[thres_order])
        })
    return pd.DataFrame.from_dict(performance)

def impute_data(train_index, data, groups, strategy = 'Median', add_missing = False, add_group = False, complete_case = False, max_iter = 10):
    """
        Imputes data given the different strategy.

        Args:
            train_index (Array): Index of the training data (used to compute median or train regressor).
            data (DataFrame): Dataset to impute (with missing data).
            groups (Array): Group membership of each point.
            strategy (str, optional): Strategy to use (MICE or Median). Defaults to 'Median'.
            add_missing (bool, optional): Add missing indicators to the returned dataset. Defaults to False.
            add_group (bool, optional): Add group to the returned dataset. Defaults to False.
            complete_case (bool, optional): Remove all individuals with missing data. Defaults to False.
            max_iter (int, optional): Iterations to use for MICE. Defaults to 10.

        Returns:
            DataFrame, Array: Imputed data, Index to use for training
    """
    index = data.loc[train_index].dropna().index if complete_case else train_index # For complete case analysis -- keep only index of complete case
    data = data.add_suffix('_data')

    if add_missing:
        # Add missing indicator
        data = pd.concat([data, data.isna().add_suffix('_missing')], axis = 1)

    if add_group:
        # Add group befoer splitting only for imputation
        data = pd.concat([data, groups.rename('Group')], axis = 1)

    if 'Group' in strategy:
        if 'Mean' in strategy:
            data = data.groupby(groups).transform(lambda x: x.fillna(x.mean()))
        elif 'Median' in strategy:
            data = data.groupby(groups).transform(lambda x: x.fillna(x.median()))
        else:
            # Add group befoer splitting only for imputation
            data = pd.concat([data, groups.rename('Group')], axis = 1)


    # Data to use to learn imputation
    train_data = data.loc[train_index]

    # MICE Algorithm
    ## 1. Init with median imputation
    imputed = pd.DataFrame(SimpleImputer(strategy = "mean" if "Mean" in strategy else "median").fit(train_data).transform(data), index = data.index, columns = data.columns)

    if 'MICE' in strategy:
        missing = data.isna()

        ## 2. Iterate through columns
        ### Find columns with random values (start with the one with least)
        to_impute = missing.sum().sort_values()
        to_impute = to_impute[to_impute > 0]

        ### Impute one by one with regression until convergence
        for _ in range(max_iter):
            for c in to_impute.index:
                #### Take train points for which c is observed to train model
                train_data = imputed.loc[train_index][~missing.loc[train_index].loc[:, c]]

                #### Fit regression
                lr = LinearRegression().fit(train_data.loc[:, imputed.columns != c], train_data.loc[:, c])
                residuals = np.abs(lr.predict(train_data.loc[:, imputed.columns != c]) - train_data.loc[:, c])

                #### Draw with normal error
                imputed.loc[:, c][missing.loc[:, c]] = lr.predict(imputed.loc[:, imputed.columns != c][missing.loc[:, c]]) + np.random.normal(scale = np.std(residuals), size = missing.loc[:, c].sum())
        else:
            if 'Group' in strategy:
                # Remove the group columns of imputed data
                imputed = imputed.iloc[:, :-1]

    return imputed, index

def train_test(data, labels, groups_bin, groups, n_imputation = 1, seed = 42, **args_imputation):  
    """
        Computes the performance on a 80 - 20 % train test split

        Args:
            data (DataFrame): Dataset with missing data.
            labels (DataFrame): Associated label to predict.
            groups_bin (Array): Binary group membership.
            groups (Array): Named group membership.
            n_imputation (int, optional): Number of imputation if > 1, multiple imputation algorithm. Defaults to 1.
            seed (int, optional): For reproducibility. Defaults to 42.

        Returns:
            Frame, Array, Array: Performance, coefficients of the logistic regression, mean imputed values
    """
    predictions, coefs, mean_imputed = [], [], []

    train, test = train_test_split(data.index, test_size = 0.2, random_state = seed)
    np.random.seed(seed)
    for i in range(n_imputation):
        # Impute data
        imputed, train = impute_data(train, data, groups_bin, **args_imputation)
        modelfit = LogisticRegression(random_state = i).fit(imputed.loc[train], labels.loc[train])
        predictions.append(modelfit.predict_proba(imputed.loc[test])[:, 1])
        coefs.append(np.array([- modelfit.intercept_[0] / modelfit.coef_[0][1], - modelfit.coef_[0][0] / modelfit.coef_[0][1]]))
        mean_imputed.append(imputed)

    predictions = np.mean(predictions, axis = 0)
    coefs = np.mean(coefs, axis = 0)
    mean_imputed = pd.concat({"Mean": pd.DataFrame(np.mean(mean_imputed, axis = 0), index = data.index, columns = imputed.columns), "Std":pd.DataFrame(np.std(mean_imputed, axis = 0), index = data.index, columns = imputed.columns)}, axis = 1)
    
    return evaluate(labels.loc[test], groups.loc[test], predictions), coefs, mean_imputed

def k_experiment(majority_size, ratio, class_balance, removal, k = 10, n_imputation = 1, **args_imputation):
    """
        Generates k datasets and computes model performance

        Returns performance over every fold
        Coef and data of the last fold

        Args:
            majority_size (int): Number of points in the majority class.
            ratio (float): Ratio in the minority class.
            class_balance (float): Ratio between positive and negative.
            removal (function): Function used to remove datapoints.
            k (int, optional): Number of experiments to run. Defaults to 10.
            n_imputation (int, optional): Number iteration of the imputaiton to use. Defaults to 1.

        Returns:
            Dict, tuplet: Performance, LAST logitic and imputation characteristics
    """
    performances = {}
    for i in trange(k):
        data, labels, protected_binarized, protected = generate_data_linear_shift(majority_size, ratio, class_balance, seed = i)
        data = removal(data, labels, protected, seed = i)
        performances[i], coefs, imputed = train_test(data, labels, protected_binarized, protected, n_imputation, seed = i, **args_imputation)

    performances = pd.concat(performances, axis = 0)
    performances.index.set_names(['Fold', 'Metric'], inplace = True)

    return performances, coefs, (data, imputed, labels, protected_binarized, protected)