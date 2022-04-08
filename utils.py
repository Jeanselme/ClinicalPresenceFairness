import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale = 1.5)

### Display functions
def display_data(data, labels, protected, colors = ['orange', 'blue'], legend = True, distribution = False):
    # Iterate through group and display data
    for p, name, color in zip([0, 1], ['Majority', 'Minority'], colors):
        data_p, label_p = data[protected == p], labels[protected == p]

        if distribution:
            sns.kdeplot(x = data_p[:, 0], y = data_p[:, 1],c = color)
        else:
            plt.scatter(data_p[:, 0][~label_p], data_p[:, 1][~label_p], alpha = 0.25, label = name, c = color)
            plt.scatter(data_p[:, 0][label_p], data_p[:, 1][label_p], alpha = 0.25, marker = 'x', c = color)

    # Formatting
    plt.scatter([],[], marker = 'x', label = 'Positive', c = 'k')
    plt.scatter([],[], marker = 'o', label = 'Negative', c = 'k')
    plt.axvline(0.5, c = 'orange', linestyle=(0, (5, 5)))
    plt.axhline(0.5, c = "blue",linestyle=(0, (5, 5)))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)

    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def display_result(min_perf, maj_perf, tot_perf):
    std = pd.DataFrame({
        'Minority': [1.96 * np.std(min_perf[i]) / np.sqrt(len(min_perf[i])) for i in min_perf],
        'Majority': [1.96 * np.std(maj_perf[i]) / np.sqrt(len(min_perf[i])) for i in min_perf],
        'Total': [1.96 * np.std(tot_perf[i]) / np.sqrt(len(min_perf[i])) for i in min_perf]
    }, index = [i for i in min_perf])

    ax = pd.DataFrame({
        'Minority': [np.mean(min_perf[i]) for i in min_perf],
        'Majority': [np.mean(maj_perf[i]) for i in min_perf],
        'Total': [np.mean(tot_perf[i]) for i in min_perf]
    }, index = [i for i in min_perf]).T.plot.barh(xerr = std.T, color = ['tab:green', 'tab:olive'])
    plt.grid(alpha = 0.3)
    plt.xlim(0.2, 1.0)
    plt.axvline(0.5, ls = ':', c = 'k', alpha = 0.5)
    plt.xlabel('ROC')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

### Modelling function
def cross_validation(data, labels, groups, folds = 5, model = LogisticRegression):
    predictions, coefs = [], []

    for (train, test) in KFold(folds, shuffle = True, random_state = 0).split(np.arange(len(data))):
        modelfit = model().fit(data[train], labels[train])
        predictions.append(pd.DataFrame({
                                'Predictions': modelfit.predict_proba(data[test])[:, 1],
                                'Truth': labels[test], 
                                'Protected': groups[test]}))
        coefs.append((- modelfit.intercept_[0] / modelfit.coef_[0][1], - modelfit.coef_[0][0] / modelfit.coef_[0][1]))           


    # Performances computation
    min_perf, maj_perf, tot_perf, min_rocs = [], [], [], []
    for pred in predictions:
        maj_perf.append(roc_auc_score(pred[~pred.Protected].Truth, pred[~pred.Protected].Predictions))
        min_perf.append(roc_auc_score(pred[pred.Protected].Truth, pred[pred.Protected].Predictions))
        tot_perf.append(roc_auc_score(pred.Truth, pred.Predictions))
        fpr, tpr, _ = roc_curve(pred.Truth, pred.Predictions)
        min_rocs.append(np.interp(np.linspace(0, 1, 1000), fpr, tpr))

    return min_perf, maj_perf, tot_perf, min_rocs, coefs

def impute_data(data, groups, strategy = 'Population'):
    if strategy == 'Population':
        return np.nan_to_num(data, nan = np.nanmedian(data[:, 0])) 

    if strategy == 'Group':
        fill = data.copy()
        for g in np.unique(groups):
            fill[groups == g] = np.nan_to_num(fill[groups == g], nan = np.nanmedian(fill[groups == g][:, 0])) 
        return fill