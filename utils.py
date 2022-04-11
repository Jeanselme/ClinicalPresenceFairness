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
            sns.kdeplot(x = data_p[:, 0], y = data_p[:, 1], color = color, alpha = 0.75)
        else:
            plt.scatter(data_p[:, 0][~label_p], data_p[:, 1][~label_p], alpha = 0.25, label = name, c = color)
            plt.scatter(data_p[:, 0][label_p], data_p[:, 1][label_p], alpha = 0.25, marker = 'x', c = color)

    # Formatting
    plt.scatter([],[], marker = 'x', label = 'Positive', c = 'k')
    plt.scatter([],[], marker = 'o', label = 'Negative', c = 'k')
    plt.axvline(0.5, c = 'orange', linestyle=(0, (5, 5)))
    plt.axhline(0.5, c = "blue", linestyle=(0, (5, 5)))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)

    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def display_result(min_perf, maj_perf, tot_perf, type = 'AUC'):
    std = pd.DataFrame({
        'Minority': [1.96 * np.std(min_perf[i]) / np.sqrt(len(min_perf[i])) for i in min_perf],
        'Majority': [1.96 * np.std(maj_perf[i]) / np.sqrt(len(min_perf[i])) for i in min_perf],
        'Overall': [1.96 * np.std(tot_perf[i]) / np.sqrt(len(min_perf[i])) for i in min_perf]
    }, index = [i for i in min_perf])

    ax = pd.DataFrame({
        'Minority': [np.mean(min_perf[i]) for i in min_perf],
        'Majority': [np.mean(maj_perf[i]) for i in min_perf],
        'Overall': [np.mean(tot_perf[i]) for i in min_perf]
    }, index = [i for i in min_perf]).T.plot.barh(xerr = std.T, color = ['tab:green', 'tab:olive'])
    plt.grid(alpha = 0.3)
    
    if type == 'AUC':
        plt.xlim(0.2, 1.0)
        plt.xlabel('AUC-ROC')
        plt.axvline(0.5, ls = ':', c = 'k', alpha = 0.5)
    else:
        plt.xlim(0., 1.0)
        plt.xlabel('Brier Score')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
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
    min_perf, maj_perf, tot_perf, maj_bier, min_bier, tot_bier, min_rocs = [], [], [], [], [], [], []
    for pred in predictions:
        maj_perf.append(roc_auc_score(pred[~pred.Protected].Truth, pred[~pred.Protected].Predictions))
        min_perf.append(roc_auc_score(pred[pred.Protected].Truth, pred[pred.Protected].Predictions))
        tot_perf.append(roc_auc_score(pred.Truth, pred.Predictions))
        maj_bier.append(brier_score_loss(pred[~pred.Protected].Truth, pred[~pred.Protected].Predictions))
        min_bier.append(brier_score_loss(pred[pred.Protected].Truth, pred[pred.Protected].Predictions))
        tot_bier.append(brier_score_loss(pred.Truth, pred.Predictions))
        fpr, tpr, _ = roc_curve(pred.Truth, pred.Predictions)
        min_rocs.append(np.interp(np.linspace(0, 1, 1000), fpr, tpr))

    return min_perf, maj_perf, tot_perf, min_rocs, maj_bier, min_bier, tot_bier, coefs

def impute_data(data, groups, strategy = 'Population'):
    if strategy == 'Population':
        return IterativeImputer(random_state = 0, max_iter = 50, imputation_order = 'random', initial_strategy = 'median').fit_transform(data)

    if strategy == 'Group':
        return IterativeImputer(random_state = 0, max_iter = 50, imputation_order = 'random', initial_strategy = 'median').fit_transform(np.concatenate([data, groups.reshape((-1, 1))], 1))[:, :-1]

from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def from_array_to_df(covariates, time, event):
    data = pd.DataFrame(covariates).copy()
    data['Time'] = time
    data['Event'] = event
    return data

class ToyExperiment():

    def train(self, *args):
        print("Toy Experiment - Results already saved")

class Experiment():

    def __init__(self, model = 'joint', hyper_grid = None, n_iter = 100, 
                random_seed = 0, times = [1, 7, 14, 30], normalization = True, path = 'results', save = True):
        self.model = model
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        self.times = times
        
        self.iter = 0
        self.best_nll = np.inf
        self.best_hyper = None
        self.best_model = None
        self.normalization = normalization
        self.path = path
        self.tosave = save

    @classmethod
    def create(cls, model = 'log', hyper_grid = None, n_iter = 100, 
                random_seed = 0, times = [1, 7, 14, 30], path = 'results', normalization = True, force = False, save = True):
        print(path)
        if not(force):
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    obj = cls.load(path + '.pickle')
                    obj.times = times
                    return obj
                except:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass
                
        return cls(model, hyper_grid, n_iter, random_seed, times, normalization, path, save)

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        return pickle.load(file)

    @staticmethod
    def save(obj):
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')
                
    def save_results(self, predictions, used):
        res = pd.concat([predictions, used], axis = 1)
        
        if self.tosave:
            res.to_csv(open(self.path + '.csv', 'w'))

        return res

    def train(self, covariates, event, training):
        """
            Model is selected with train / test split and maximum likelihood

            Args:
                covariates (Dataframe n * d): Observed covariates
                event (Dataframe n): Event indicator
                training (Dataframe n): Indicate which points should be used for training

            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        # Split source domain into train, test and dev
        all_training = training[training].index
        training_index, test_index = train_test_split(all_training, train_size = 0.9, 
                                            random_state = self.random_seed)
        training_index, validation_index = train_test_split(training_index, train_size = 0.9, 
                                            random_state = self.random_seed)
        annotated_training = pd.Series("Train", training.index, name = "Use")
        annotated_training[test_index] = "Internal"
        annotated_training[~training] = "External"

        train_cov, train_event = covariates.loc[training_index], event.loc[training_index]
        dev_cov, dev_event = covariates.loc[validation_index], event.loc[validation_index]
        
        if self.normalization:
            self.normalizer = StandardScaler().fit(train_cov)
            train_cov = pd.DataFrame(self.normalizer.transform(train_cov), index = train_cov.index)
            dev_cov = pd.DataFrame(self.normalizer.transform(dev_cov), index = dev_cov.index)
            covariates = pd.DataFrame(self.normalizer.transform(covariates), index = covariates.index)

        # Train on subset one domain
        ## Grid search best params
        for i, hyper in enumerate(self.hyper_grid):
            if i < self.iter:
                # When object is reloaded - Avoid to recompute same parameters
                continue
            model = self._fit(train_cov, train_event, hyper)

            if model:
                nll = self._nll(model, dev_cov, dev_event)
                if nll < self.best_nll:
                    self.best_hyper = hyper
                    self.best_model = model
                    self.best_nll = nll

            self.iter += 1
            Experiment.save(self)

        return self.save_results(self.predict(covariates, training.index), annotated_training)

    def predict(self, covariates, index = None):
        if self.best_model is None:
            raise ValueError('Model not trained - Call .fit')
        else:
            predictions = pd.DataFrame(self.best_model.predict_proba(covariates), index = index)
        return predictions
            
    def _fit(self, covariates, event, hyperparameter):
        np.random.seed(self.random_seed)
        if self.model == "log":
            model = LogisticRegression(**hyperparameter)
            return model.fit(covariates, event)
        else:
            raise ValueError('Model {} unknown'.format(self.model))
        
    def _nll(self, model, covariates, event):
        return - model.score(covariates, event)