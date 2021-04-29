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
import daal4py as d4p
from boostember import *
from boostember.features_extended import *
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve
from sklearnex import patch_sklearn
from yellowbrick.model_selection import LearningCurve
patch_sklearn()

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

scorecheck = {
        'roc_auc': make_scorer(roc_auc_score, max_fpr=5e-3),
        'precision': make_scorer(average_precision_score),
        'accuracy': make_scorer(accuracy_score)
        }

nsplit=3

with mlflow.start_run(run_name="demo_ember_lightgbm_lc_cpuboost") as run:
    mlflow.set_tags({"description": "Demo","model": "lightgbm","summary": f"lc with {nsplit} cpu boost"})
    lgbm_dataset = lgb.Dataset(X_train, y_train)
    lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', n_jobs=int(multiprocessing.cpu_count()/4))


    cv_ember = LearningCurve(estimator=lgbm_model, scoring=make_scorer(accuracy_score), shuffle=True, cv=TimeSeriesSplit(n_splits=nsplit).split(X_train), n_jobs=1, verbose=10)
    cv_ember.fit(X_train, y_train)
    cv_ember.show(outpath='learningcurve.png')
    mlflow.log_param('n_split',nsplit)
    joblib.dump(lgbm_model, 'model.pkl')
    mlflow.log_artifact('model.pkl', 'additional')
    mlflow.log_artifact('learningcurve.png', 'additional')
    print(dir(cv_ember))
    #mlflow.log_dict(cv_ember.best_params_, 'best_params.json')
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

