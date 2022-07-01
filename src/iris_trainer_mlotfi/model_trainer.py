import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from functools import partial

from typing import Tuple, Dict

import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    trials: Trials()
    best_hyperparams: Dict
    model: ClassifierMixin
    hyperparam_space: Dict
    possible_values: Dict

    def __init__(self, model_class: ClassifierMixin, performance_metric, data, multi_class=None):
        """ Class to train a supervised model using a specific metric.

        :param model_class: Class of a sklearn-style classifier. Should not be an instance, but a class.
        :param performance_metric: Performance metric to optimize hyperparameters on.
        :param data: Training and test data as tuple: (X_train, X_test, y_train, y_test).
        :param n_splits: Number of splits for the k-fold cross validation.
        :param plot_directory: Folder to write the plots to.
        """
        self.model_class = model_class
        if multi_class is not None:
            self.performance_metric = partial(performance_metric, multi_class=multi_class)
        else:
            self.performance_metric = performance_metric
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.trials = Trials()


    def add_possible_hyperparameters(self, space, possible_values={}):
        for k, v in possible_values.items():
            space.update({k: hp.choice(k, v)})

        self.hyperparam_space = space
        self.possible_values = possible_values

    def run_hyperparameter_search(self, number_of_trials:int = 100, alternative_obj: bool = False, n_splits : int=3):
        """ Runs the hyperparameter search on hyperparameter space for a specified number of times.

        :param hyperparameter_space: Hyperopt hyperparameter space to use..
        :param number_of_trials: Number of iterations to run for
        :param alternative_obj: If true, an objective that penalizes overfitting is used.
        :return: best parameters.
        """
        hyperparameter_space = self.hyperparam_space
        data = (self.X_train, self.y_train)
        current_obj = partial(self.objective, data=data,
                              performance_metric=self.performance_metric, model_class=self.model_class,
                              alternative_obj=alternative_obj, n_splits=n_splits)
        argmin = fmin(fn=current_obj,
                      space=hyperparameter_space,
                      algo=tpe.suggest,
                      max_evals=number_of_trials,
                      trials=self.trials
                      )

        for param_name in self.possible_values.keys():
            argmin[param_name] = self.possible_values[param_name][argmin[param_name]]

        self.best_hyperparams = argmin

        return argmin

    def train_final_model(self, best_hyperparams=None):
        """ Trains final model. If not provided, best hyperparameters saved during the last search are used.

        :param best_hyperparams:
        :return: Dict with model and performance information.
        """
        if best_hyperparams is None:
            best_hyperparams = self.best_hyperparams

        self.model = self.model_class(**best_hyperparams)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict_proba(self.X_test)
        y_pred_train = self.model.predict_proba(self.X_train)
        performance_train = self.performance_metric(self.y_train, y_pred_train)
        performance_val = self.performance_metric(self.y_test, y_pred)

        print(f' train: {performance_train}, test: {performance_val}')
        print(self.best_hyperparams)

        if isinstance(self.performance_metric, partial):
            metric_name = self.performance_metric.func.__name__
        else:
            metric_name = self.performance_metric.__name__

        return self.model, {
            f'{metric_name}_train': performance_train,
            f'{metric_name}_val': performance_val,
            'overfitting': performance_train - performance_val
        }

    @staticmethod
    def objective(args, data, alternative_obj: bool, model_class, performance_metric, n_splits):
        """ Objective function for hyperparameter search.

        :param args: Hyperparameters for the model class
        :param data: Data for training Xtrain, Xval, y_train, y_val
        :param folds: dataframe with fold information by clients.
        :param alternative_obj: If true, the objective function is modified to penalize overfitting.
        :param model_class: model class to train.
        :param performance_metric: performance metric to optimize (e.g. average_precision_score).
        :return: objective value and extra performance information for hyperopt.
        """
        X, y = data

        performance = []
        val_metric = []
        train_metric = []
        overfitting = []
        strat_kfold_iterator = StratifiedKFold(n_splits=n_splits)
        for fold, (train_index, test_index) in enumerate(strat_kfold_iterator.split(X, y)):
            X_train = X[train_index, :]
            X_val = X[test_index, :]
            y_train = y[train_index]
            y_val = y[test_index]

            model = model_class(**args)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)
            y_pred_train = model.predict_proba(X_train)
            metric_train = performance_metric(y_train, y_pred_train)
            metric_val = performance_metric(y_val, y_pred)

            if alternative_obj:
                result_fold = metric_val ** 2 * (1 - (metric_train - metric_val))
            else:
                result_fold = metric_val

            performance.append(result_fold)
            val_metric.append(metric_val)
            train_metric.append(metric_train)
            overfitting.append((metric_train - metric_val))

        obj = np.mean(performance)

        return {'loss': -obj,
                'train_metric': np.mean(train_metric),
                'val_metric': np.mean(val_metric),
                'overfitting': np.mean(overfitting),
                'status': STATUS_OK}
