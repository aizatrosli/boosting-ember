import sys,os,time,platform,multiprocessing
import joblib,shap,mlflow
import pandas as pd
import numpy as np

datasetpath = '/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018'
boostingpath ='/home/aizat/OneDrive/Master Project/Workspace/boosting-ember'
sys.path.append('/home/aizat/ember')
sys.path.append(boostingpath)
import matplotlib.pyplot as plt
import xgboost as xgb
import ember
import boostember
import daal4py as d4p
from boostember import *
from boostember.features_extended import *
from sklearn.metrics import plot_roc_curve,auc
from sklearn.utils import shuffle
from sklearnex import patch_sklearn
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
mlflow.xgboost.autolog()

nsplit=3
params = {
        'booster':['dart'],
        'objective':['binary:logistic'],
        }

scorecheck = {
    'roc_auc': make_scorer(roc_auc_score, max_fpr=5e-3),
    'precision': make_scorer(average_precision_score),
    'accuracy': make_scorer(accuracy_score)
}

disparr = []

with mlflow.start_run(run_name="demo_ember_xgboost_cpuboost") as run:
    mlflow.set_tags({"description": "Demo","model": "xgboost","summary": f"cv with {nsplit} cpu boost"})
    starttime = time.time()
    model = xgb.XGBClassifier(booster="dart", objective="binary:logistic", n_jobs=-2)
    model.fit(X_train, y_train)
    mlflow.log_metric('fit_time', time.time()-starttime)
    mlflow.log_param('n_split', nsplit)
    mlflow.sklearn.log_model(model, 'skmodel')
    mlflow.xgboost.log_model(model.get_booster(), 'xgbmodel')
    mlflow.log_params(model.get_params())
    for ix, (train_ix, test_ix) in enumerate(TimeSeriesSplit(n_splits=nsplit).split(X_train)):
        with mlflow.start_run(run_name=f'cross_validation_{ix}', nested=True) as child_run:
            cvmodel = model.copy()
            cvtime = time.time()
            cvmodel.fit(X_train[train_ix], y_train[train_ix])
            mlflow.log_metric('fit_time', time.time() - cvtime)
            disp = plot_roc_curve(cvmodel, X_test, y_test)
            disparr.append(disp)
            mlflow.log_figure(disp.figure_, f"{ix}_plot_roc_curve.png")
            mlflow.sklearn.eval_and_log_metrics(model=cvmodel, X=X_test, y_true=y_test, prefix=f'{ix}')
            mlflow.sklearn.log_model(cvmodel, 'skmodel')
            mlflow.xgboost.log_model(cvmodel.get_booster(), 'xgbmodel')
            mlflow.log_params(cvmodel.get_params())
    mlflow.log_metric('validation_time', time.time() - starttime)
    #joblib.dump(model.best_estimator_, 'estimator.pkl')
    #mlflow.log_artifact('estimator.pkl', 'raw')
    #mlflow.log_dict(model.best_params_, 'best_params.json')
    #mlflow.log_dict(model.cv_results_, 'cv_results.json')
    for key,data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        print(data)
        # show data logged in the child runs
        filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run.info.run_id)
        runs = mlflow.search_runs(filter_string=filter_child_runs)
        param_cols = ["params.{}".format(p) for p in params.keys()]
        metric_cols = ["metrics.mean_test_score"]
        print("\n========== end ==========\n")

    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    for disp in disparr:
        interp_tpr = np.interp(mean_fpr, disp.fpr, disp.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(disp.roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    mlflow.log_figure(fig, f"plot_roc_curve.png")