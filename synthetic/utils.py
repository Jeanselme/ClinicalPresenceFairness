# Contains all the function for the synthetic experiements

# Import all libraries
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors

import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False,
                 "axes.spines.bottom": False, "figure.dpi": 300, 'savefig.dpi': 300}
sns.set_theme(style = "whitegrid", rc = custom_params, font_scale = 1.75)


from sklearn.impute import SimpleImputer, KNNImputer
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
            sns.kdeplot(x = data_p.iloc[:, 1][~label_p], y = data_p.iloc[:, 0][~label_p], color = color, alpha = 0.7, linestyles = 'dashed', linewidth = 2, levels = 5)
            sns.kdeplot(x = data_p.iloc[:, 1][label_p], y = data_p.iloc[:, 0][label_p], color = color, alpha = 0.7, linestyles = 'dotted', linewidth = 2, levels = 5)
        else:
            plt.scatter(data_p.iloc[:, 1][~label_p], data_p.iloc[:, 0][~label_p], alpha = 0.25, label = name, c = color)
            plt.scatter(data_p.iloc[:, 1][label_p], data_p.iloc[:, 0][label_p], alpha = 0.25, marker = 'x', c = color)

    # Formatting
    plt.scatter([],[], label = 'Group', alpha = 0)
    plt.scatter([],[], label = 'Marginalised', c = 'tab:blue', alpha = 0.7)
    plt.scatter([],[], label = 'Majority', c = 'tab:orange', alpha = 0.7)
    plt.scatter([],[], label = ' ', alpha = 0)
    plt.scatter([],[], label = 'Class', alpha = 0)
    plt.plot([],[], ls = 'dotted', label = 'Positive', c = 'k')
    plt.plot([],[], ls = 'dashed', label = 'Negative', c = 'k')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    # plt.xlim(-1, 2)
    # plt.ylim(-1, 2)

    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def print_pandas_latex(mean, std):
    print(pd.DataFrame.from_dict({m: ["{:.3f} ({:.3f})".format(mean.loc[m].loc[i], std.loc[m].loc[i]) for i in mean.columns] for m in mean.index}, columns = mean.columns, orient = 'index').to_latex())

def display_result(performance, type = 'AUC', legend = True, colors = ['tab:orange', 'tab:blue', 'tab:gray'], alphas = None, plot = False):
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
    mean, std, ci = {}, {}, {}
    for method in performance:
        mean[method], std[method], ci[method] = {}, {}, {}
        for group in performance[method].columns:
            assert len(performance[method].columns) == len(colors), 'Not enough colors provided'
            meth_group = performance[method][group]
            meth_group = meth_group[meth_group.index.get_level_values('Metric') == type]
            mean[method][group] = meth_group.mean()
            std[method][group] = meth_group.std()
            ci[method][group] = 1.96 * meth_group.std() / np.sqrt(len(meth_group))

    mean, std, ci = pd.DataFrame.from_dict(mean), pd.DataFrame.from_dict(std), pd.DataFrame.from_dict(ci)
    print_pandas_latex(mean, std)

    if not plot: return
    ax = mean.plot.barh(xerr = ci, legend = legend, figsize = (7, 7))
    # Change colors
    hatches = ['', 'ooo', 'xx', '//', '||', '***', '++'] * 2
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

def generate_data_linear_corr_shift(majority_size, ratio, class_balance = 0.5, seed = 0):
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
    concatenation[:, 0] += 0.5 * concatenation[:, 1]

    labels = np.concatenate([labels_maj, labels_min])
    protected =  np.concatenate([np.full(len(labels_maj), False), np.full(len(labels_min), True)])

    sort = np.arange(len(concatenation))
    np.random.shuffle(sort)
    return pd.DataFrame(concatenation[sort]), pd.Series(labels[sort]), pd.Series(protected[sort]), pd.Series(protected[sort]).replace({True: 'Minority', False: 'Majority'})


def generate_data_same(majority_size, ratio, class_balance = [0.1, 0.5], seed = 0):
    """
        Generate data with similar data distributionbut difference in disease prevalence

        Args:
            majority_size (int): Number points for majority
            ratio (float): Ratio for minority (0.1 = 1 in minority for 10 in majority)
            class_balance (float, optional): Balance between class. Default 0.5.
    """
    np.random.seed(seed)

    majority_pos = 0.25 * np.random.randn(int(majority_size * class_balance[0]), 2)
    majority_neg = 0.25 * np.random.randn(int(majority_size * (1 - class_balance[0])), 2)
    majority_pos += 0.5
    majority = np.concatenate([majority_pos, majority_neg])
    labels_maj = np.concatenate([np.full(len(majority_pos), True), np.full(len(majority_neg), False)])

    minority_pos = 0.25 * np.random.randn(int(majority_size * ratio * class_balance[1]), 2)
    minority_neg = 0.25 * np.random.randn(int(majority_size * ratio * (1 - class_balance[1])), 2)
    minority_pos += 0.5
    minority = np.concatenate([minority_pos, minority_neg])
    labels_min = np.concatenate([np.full(len(minority_pos), True), np.full(len(minority_neg), False)])

    concatenation = np.concatenate([majority, minority])
    labels = np.concatenate([labels_maj, labels_min])
    protected =  np.concatenate([np.full(len(labels_maj), False), np.full(len(labels_min), True)])

    sort = np.arange(len(concatenation))
    np.random.shuffle(sort)
    return pd.DataFrame(concatenation[sort]), pd.Series(labels[sort]), pd.Series(protected[sort]), pd.Series(protected[sort]).replace({True: 'Minority', False: 'Majority'})

### Modelling function
def evaluate(y_true, groups, y_pred, p = 0.3):
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

    threshold_top = pd.Series(y_pred).nlargest(int(len(y_pred) * p), keep = 'all').min()

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
        selected = y_pred_group >= threshold_top

        performance[group] = pd.Series({
            "Brier Score": brier_score_loss(y_true_group, y_pred_group),
            "AUC": roc_auc_score(y_true_group, y_pred_group),
            
            "FPR @ 90% TPR": np.interp(threshold_fpr, thresholds[thres_order], fpr[thres_order]),
            "TPR @ 10% FPR": np.interp(threshold_tpr, thresholds[thres_order], tpr[thres_order]),

            "Wrongly prioritised": (1 - y_true_group[selected]).sum() / (1 - y_true_group).sum(),
            "% of non-prioritized high-risk patients (FNR)": (y_true_group[~selected]).sum() / y_true_group.sum(),
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
    missing = data.isna()

    # Data to use to learn imputation
    train_data = data.loc[train_index]
    imputed = data.copy()

    if 'Hot Deck' in strategy:
        imputer = KNNImputer(n_neighbors=1, weights="uniform")
        if 'Group' in strategy:
            for group in np.unique(groups):
                train_group = train_data[groups.loc[train_index] == group]
                imputer.fit(train_group)
                imputed[groups == group] = imputer.transform(data[groups == group])
        else:
            imputer.fit(train_data)
            imputed = pd.DataFrame(imputer.transform(data), index = data.index, columns = data.columns)
        

    if 'Mean' in strategy:
        if 'Group' in strategy:
            mean_group = train_data.groupby(groups.loc[train_index]).mean()
            imputed = data.groupby(groups).transform(lambda x: x.fillna(mean_group.loc[groups.loc[x.index[0]]][x.name]))
        imputed = imputed.fillna(train_data.mean())

    # MICE Algorithm
    ## 1. Init with mean imputation
    if 'MICE' in strategy:
        imputed = pd.DataFrame(SimpleImputer(strategy = "mean").fit(train_data).transform(data), index = data.index, columns = data.columns)
        if "Group" in strategy:
            data = pd.concat([data, groups.rename('Group')], axis = 1)
            imputed = pd.concat([imputed, groups.rename('Group')], axis = 1)
            train_data = data.loc[train_index]
            
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

    if add_missing:
        # Add missing indicator
        imputed = pd.concat([imputed, missing.add_suffix('_missing').astype(float)], axis = 1)

    if add_group:
        # Add group befoer splitting only for imputation
        imputed = pd.concat([imputed, groups.rename('Group').astype(float)], axis = 1)

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
        modelfit = LogisticRegression(random_state = i, penalty = None, max_iter = 1000).fit(imputed.loc[train], labels.loc[train])
        predictions.append(modelfit.predict_proba(imputed.loc[test])[:, 1])
        coefs.append(np.array([- modelfit.intercept_[0] / modelfit.coef_[0][1], - modelfit.coef_[0][0] / modelfit.coef_[0][1]]))
        mean_imputed.append(imputed)

    predictions = np.mean(predictions, axis = 0)
    coefs = np.mean(coefs, axis = 0)
    mean_imputed = pd.concat({"Mean": pd.DataFrame(np.mean(mean_imputed, axis = 0), index = data.index, columns = imputed.columns), "Std":pd.DataFrame(np.std(mean_imputed, axis = 0), index = data.index, columns = imputed.columns)}, axis = 1)
    
    return evaluate(labels.loc[test], groups.loc[test], predictions), coefs, mean_imputed

def k_experiment(majority_size, ratio, class_balance, removal, k = 10, n_imputation = 1, generate = generate_data_linear_shift, **args_imputation):
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
    delta_reconstruction = {"Overall": [], "Minority": [], "Majority": [], "Difference": []}
    mean_observed = {"Overall": [], "Minority": [], "Majority": []}
    obs_rate = {"Overall": [], "Minority": [], "Majority": []}
    corr_missingness = {"Overall": [], "Minority": [], "Majority": []}
    corr_covariates = {"Overall": [], "Minority": [], "Majority": []}
    for i in trange(k):
        data, labels, protected_binarized, protected = generate(majority_size, ratio, class_balance, seed = i)
        data_removed = removal(data, labels, protected, seed = i)
        corr_covariates["Overall"].append((data_removed).corr().iloc[0, 1])
        corr_covariates["Minority"].append((data_removed[protected_binarized]).corr().iloc[0, 1])
        corr_covariates["Majority"].append((data_removed[~protected_binarized]).corr().iloc[0, 1])

        performances[i], coefs, imputed = train_test(data_removed, labels, protected_binarized, protected, n_imputation, seed = i, **args_imputation)

        error = (data.iloc[:, 0] - imputed.Mean.values[:, 0])**2# Select dimension 0 as it is the one where we removed data
        error_min = error.loc[protected_binarized].loc[data_removed[protected_binarized].isna().values]
        error_maj = error.loc[~protected_binarized].loc[data_removed[~protected_binarized].isna().values]
        error_min = 0 if error_min.empty else error_min.mean()
        error_maj = 0 if error_maj.empty else error_maj.mean()

        delta_reconstruction["Overall"].append(error.loc[data_removed.isna().values].mean())
        delta_reconstruction["Minority"].append(error_min)
        delta_reconstruction["Majority"].append(error_maj)
        delta_reconstruction["Difference"].append(error_min - error_maj)

        mean_observed["Overall"].append(data_removed.dropna().mean().iloc[0])
        mean_observed["Minority"].append(data_removed.loc[protected_binarized].dropna().mean().iloc[0])
        mean_observed["Majority"].append(data_removed.loc[~protected_binarized].dropna().mean().iloc[0])

        obs_rate["Overall"].append(data_removed.isna().mean().iloc[0])
        obs_rate["Minority"].append(1 - data_removed.loc[protected_binarized].isna().mean().iloc[0])
        obs_rate["Majority"].append(1 - data_removed.loc[~protected_binarized].isna().mean().iloc[0])

        corr = pd.concat([data.iloc[:, 0], ~data_removed.iloc[:, 0].isna().astype(int)], axis = 1)
        corr_missingness["Overall"].append(corr.corr().min().values[0])
        corr_missingness["Minority"].append(corr.loc[protected_binarized].corr().min().values[0])
        corr_missingness["Majority"].append(corr.loc[~protected_binarized].corr().min().values[0])


    performances = pd.concat(performances, axis = 0)
    performances.index.set_names(['Fold', 'Metric'], inplace = True)

    delta_reconstruction = pd.concat({"Mean": pd.DataFrame.from_dict(delta_reconstruction).mean(), "Std": pd.DataFrame.from_dict(delta_reconstruction).std()})
    mean_observed = pd.concat({"Mean": pd.DataFrame.from_dict(mean_observed).mean(), "Std": pd.DataFrame.from_dict(mean_observed).std()})
    obs_rate = pd.concat({"Mean": pd.DataFrame.from_dict(obs_rate).mean(), "Std": pd.DataFrame.from_dict(obs_rate).std()})
    corr_missingness = pd.concat({"Mean": pd.DataFrame.from_dict(corr_missingness).mean(), "Std": pd.DataFrame.from_dict(corr_missingness).std()})

    return performances, coefs, (delta_reconstruction, mean_observed, obs_rate, corr_missingness, corr_covariates), (data, imputed, labels, protected_binarized, protected)