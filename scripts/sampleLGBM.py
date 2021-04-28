import sys,os,time,platform,multiprocessing
import joblib,shap,mlflow
import pandas as pd
import numpy as np

datasetpath = '/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018'
boostingpath ='/home/aizat/OneDrive/Master Project/Workspace/boosting-ember'
sys.path.append('/home/aizat/ember')
sys.path.append(boostingpath)

import lightgbm as lgb
import boostember
from boostember import *
from boostember.features_extended import *

boostember.mlflowsetup(os.path.join(boostingpath, 'mlflow'))


X_train, y_train, X_test, y_test = ember.read_vectorized_features(datasetpath)
delunlabel = (y_train != -1)
X_train = X_train[delunlabel]
#X_train = pd.DataFrame(X_train, columns=emberfeaturesheader())
y_train = y_train[delunlabel]
print(X_train.shape, y_train.shape)


mlflow.set_tracking_uri("https://atlascompanion.live/")
mlflow.set_experiment("Demo")
mlflow.sklearn.autolog()
mlflow.lightgbm.autolog()

with mlflow.start_run(run_name="demo_ember_lightgbm_multimetrics") as run:
    mlflow.set_tags({"description": "Demo","model": "lightgbm","summary": "multimetrics"})
    lgbm_dataset = lgb.Dataset(X_train, y_train)
    lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt', n_jobs=multiprocessing.cpu_count()/4)
    cv_ember = generic_crossvalidation(lgbm_model, X_train, y_train)
    cv.ember.fit(X_train, y_train)
    for key,data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        print(data)
        # show data logged in the child runs
        filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run.info.run_id)
        runs = mlflow.search_runs(filter_string=filter_child_runs)
        param_cols = ["params.{}".format(p) for p in params.keys()]
        metric_cols = ["metrics.mean_test_score"]
        print("\n========== child runs ==========\n")
        pd.set_option("display.max_columns", None)  # prevent truncating columns
        print(runs[["run_id", *param_cols, *metric_cols]])

