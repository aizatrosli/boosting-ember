import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib,time,os,sys,json
from sklearn.metrics import (roc_auc_score, make_scorer)
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from ember import *

def optimize_model_best(data_dir):
    """
    Run a grid search to find the best LightGBM parameters
    """
    # Read data
    X_train, y_train = read_vectorized_features(data_dir, subset="train")

    # Filter unlabeled data
    train_rows = (y_train != -1)

    # read training dataset
    X_train = X_train[train_rows]
    y_train = y_train[train_rows]

    # score by roc auc
    # we're interested in low FPR rates, so we'll consider only the AUC for FPRs in [0,5e-3]
    score = make_scorer(roc_auc_score, max_fpr=5e-3)

    # define search grid
    param_grid = {
        'boosting_type': ['goss','gdbt','dart'],
        'objective': ['binary'],
        'num_iterations': [500, 1000, 1500],
        'learning_rate': [0.005, 0.05, 0.5],
        'num_leaves': [512, 1024, 2048, 4096],
        'feature_fraction': [0.5, 0.8, 1.0, 1.3],
        'bagging_fraction': [0.5, 0.8, 1.0, 1.3]
    }
    model = lgb.LGBMClassifier(boosting_type="goss", n_jobs=-1, silent=True)

    # each row in X_train appears in chronological order of "appeared"
    # so this works for progrssive time series splitting
    progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train)

    grid = GridSearchCV(estimator=model, cv=progressive_cv, param_grid=param_grid, scoring=score, n_jobs=1, verbose=3)
    grid.fit(X_train, y_train)
    joblib.dump(grid, 'lgb_{}_{}.pkl'.format(os.path.split(data_dir)[-1],time.strftime("%Y%m%d-%H%M%S")))

    return grid.best_params_


def train_model_extended(data_dir, algo="lightgbm", params={}, feature_version=2):
    """
    Train the model from the EMBER dataset from the vectorized features.
    Extension from train_model()
    """
    # Read data
    X_train, y_train = read_vectorized_features(data_dir, "train", feature_version)
    # Filter unlabeled data
    train_rows = (y_train != -1)
    # Train
    if algo == "lightgbm":
        import lightgbm as lgb
        return lgb.train(params, lgb.Dataset(X_train[train_rows], y_train[train_rows]))
    elif algo == "catboost":
        import catboost as cat
        from catboost import Pool
        return cat.train(params, cat.Pool(X_train[train_rows], y_train[train_rows]))
    elif algo == 'xgboost':
        import xgboost as xb
        from xgboost import DMatrix
        return xb.train(params, xb.DMatrix(X_train[train_rows], y_train[train_rows]))
    return lgbm_model

'''
data_dir = '/analytics/playground/aizat/ember/dataset/ember2018'
X_train, y_train, X_test, y_test = read_vectorized_features(data_dir, feature_version=2)
params = optimize_model_best(data_dir)
print("Best parameters: ")
print(json.dumps(params, indent=2))
lgbm_model = train_model(data_dir, params, feature_version=2)
lgbm_model.save_model(os.path.join(data_dir, "ember_model_best.txt"))
'''