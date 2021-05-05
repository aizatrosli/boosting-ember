import sys,os,time,platform,multiprocessing
import joblib,shap,mlflow
import pandas as pd
import numpy as np

datasetpath = '/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018'
boostingpath ='/home/aizat/OneDrive/Master Project/Workspace/boosting-ember'
sys.path.append('/home/aizat/ember')
sys.path.append(boostingpath)

import lightgbm as lgb
import ember
import boostember
from boostember import *
from boostember.features_extended import *
from sklearn.utils import shuffle


boostember.mlflowsetup(os.path.join(boostingpath, 'mlflow'))


X_train, y_train, X_test, y_test = ember.read_vectorized_features(datasetpath)
X_train, y_train = shuffle(X_train, y_train, random_state=32)
delunlabel = (y_train != -1)
X_train = X_train[delunlabel]
#X_train = pd.DataFrame(X_train, columns=emberfeaturesheader())
y_train = y_train[delunlabel]

print(X_train.shape, y_train.shape)


mlflow.set_tracking_uri("https://atlascompanion.live/")
mlflow.set_experiment("Demo")
mlflow.sklearn.autolog()
mlflow.lightgbm.autolog()

params = {
        'boosting_type':['gbdt'],
        'objective':['binary'],
        }

nsplit=1000

with mlflow.start_run(run_name="demo_ember_lightgbm_multimetrics") as run:
    mlflow.set_tags({"description": "Demo","model": "lightgbm","summary": f"multimetrics with {nsplit}"})
    lgbm_dataset = lgb.Dataset(X_train, y_train)
    lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt', n_jobs=int(multiprocessing.cpu_count()/4))
    scorecheck = {'roc_auc': make_scorer(roc_auc_score, max_fpr=5e-3),
            'precision': make_scorer(average_precision_score),
            'accuracy': make_scorer(accuracy_score)}
    cv_ember = GridSearchCV(estimator=lgbm_model, param_grid=params, scoring=scorecheck, cv=TimeSeriesSplit(n_splits=nsplit).split(X_train), n_jobs=int(multiprocessing.cpu_count()/4), verbose=10, refit='accuracy')
    cv_ember.fit(X_train, y_train)
    mlflow.log_param('n_split',nsplit)
    joblib.dump(lgbm_model, 'model.pkl')
    mlflow.log_artifact('model.pkl', 'raw')
    joblib.dump(cv_ember.best_estimator_, 'estimator.pkl')
    mlflow.log_artifact('estimator.pkl', 'raw')
    mlflow.log_dict(cv_ember.best_params_, 'best_params.json')
    #mlflow.log_dict(cv_ember.cv_results_, 'cv_results.json')
    for key,data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        print(data)
        # show data logged in the child runs
        filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run.info.run_id)
        runs = mlflow.search_runs(filter_string=filter_child_runs)
        param_cols = ["params.{}".format(p) for p in params.keys()]
        metric_cols = ["metrics.mean_test_score"]
        print("\n========== end ==========\n")

