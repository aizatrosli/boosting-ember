import joblib, time, os, sys, json, logging, inspect, gc, multiprocessing
from copy import deepcopy
from memory_profiler import memory_usage
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


from ember import *
from .debug import *
from .features_extended import *
from .utlis import *
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from sklearnex import patch_sklearn
patch_sklearn()


class Boosting(object):

    def __init__(self, session, experiment='Demo', booster='lgbm', dataset='ember2018', features=None, shuffle=False, n_jobs=0, min_features=20, url=None, configpath='config/fyp.json'):
        import mlflow, ember
        self.startsessiontime, self.memmetrics, self.model = None, None, None
        self.shuffle = shuffle
        self.session = session
        self.experiment = experiment
        self._ember = ember
        self._mlflow = mlflow
        self.booster = booster
        self.min_features = min_features
        self.n_jobs = int(n_jobs)
        self.max_fpr = [0.01, 0.001]
        self.projectpath = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
        self.logger = self.create_logsession(session)
        self.config = self.load_config(os.path.join(self.projectpath, configpath))
        if url is None:
            self.setup_mlflow(os.path.join(self.projectpath, 'config', 'mlflow'))
        else:
            self.setup_mlflow(url)
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(dataset)
        self.y_train_pred, self.y_test_pred = None, None
        self.features = emberfeatures().features
        if features is not None and isinstance(features, list) and len(self.features) == len(self.X_train.shape[1]):
            features_ix = [self.features.index(x) for i,x in enumerate(features)]
            self.X_train, self.X_test = self.X_train[:, features_ix], self.X_test[:, features_ix]
            self.features = features

    def load_config(self, configpath):
        with open(configpath, 'rb') as fp:
            return json.load(fp)

    def setup_mlflow(self, url, file='mlflow.json'):
        import requests, pickle
        from urllib.parse import urlparse
        data = pickle.loads(requests.get(url).content) if urlparse(url).scheme else pickle.loads(open(url, 'rb').read())
        self.config.update(data)
        for key, val in data.items():
            if isinstance(val, str):
                os.environ[key] = val
        with open(file, 'w') as fp:
            json.dump(data['gcloud'], fp)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file
        self._mlflow.set_tracking_uri(self.config['tracking_uri'])
        self._mlflow.set_experiment(self.experiment)

    def create_logsession(self, session='New'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler('logfile.log')
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        return self.logger

    def params(self, estimator, stage=''):
        self.stage = [stage] if stage else []
        params = estimator.get_all_params() if self.booster == 'cb' else estimator.get_params()
        self._mlflow.log_params(params)
        self._mlflow.log_params(self.configrun)

    def metrics(self, estimator, stage=''):
        self.stage = [stage] if stage else []
        self._mlflow.log_metric(self.keyname('fit_time'), time.time()-self.startsessiontime)
        self._mlflow.sklearn.eval_and_log_metrics(model=estimator, X=self.X_test, y_true=self.y_test, prefix=f'{stage}_')
        if self.y_test_pred is not None:
            for maxfpr in self.max_fpr:
                thresh, fpr = get_threshold(self.y_test, self.y_test_pred, maxfpr)
                fnr = get_fnr(self.y_test, self.y_test_pred, thresh, 1)
                self._mlflow.log_metric(self.keyname(f'roc_auc_score_{maxfpr * 100:.4f}'), roc_auc_score(self.y_test, self.y_test_pred, max_fpr=maxfpr))
                self._mlflow.log_metric(self.keyname(f'threshold_{maxfpr * 100:.4f}'), maxfpr)
                self._mlflow.log_metric(self.keyname(f'fpr_{maxfpr * 100:.4f}'), fpr * 100)
                self._mlflow.log_metric(self.keyname(f'fnr_{maxfpr * 100:.4f}'), fnr * 100)
                self._mlflow.log_metric(self.keyname(f'detection_rate_{maxfpr * 100:.4f}'), fnr * 100)
                self._mlflow.log_figure(plot_roc(self.y_test, self.y_test_pred, fpr, fnr), f'roc_{stage}_{maxfpr}.png')
        if self.min_features:
            fi_df = pd.DataFrame(sorted(zip(estimator.feature_importances_, self.features)), columns=['Value', 'Features'])
            fi_df = fi_df.sort_values(by='Value', ascending=False)
            fi_df.to_csv(f'features_importance_{stage}.csv', index=False)
            self._mlflow.log_artifact(f'features_importance_{stage}.csv')
            fig = fi_df.head(self.min_features).plot.barh(x='Features', y='Value')
            self._mlflow.log_figure(fig.figure, f'features_importance_{stage}.png')

        if self.memmetrics:
            df = pd.DataFrame({'time (seconds)': np.linspace(0, len(self.memmetrics) * .1, len(self.memmetrics)),
                               f'Memory consumption {stage} (in MB)': self.memmetrics})
            df.to_csv('memory_training.csv', index=False)
            self._mlflow.log_metric('peak_memory_usage in MB', df[f'Memory consumption {stage} (in MB)'].max())
            self._mlflow.log_artifact('memory_training.csv')
            df = df.set_index(['time (seconds)'])
            fig = df.plot.line()
            self._mlflow.log_figure(fig.figure, 'memory_training.png')
        df = pd.DataFrame({'X_test': self.X_test, 'y_true': self.y_test, 'y_pred': self.y_test_pred})
        df.to_csv(f'testing_dataset_{stage}.csv', index=False)
        self._mlflow.log_artifact(f'testing_dataset_{stage}.csv')

    def keyname(self, name):
        return '_'.join(self.stage + [name])

    def load_data(self, datasetpath):
        if not os.path.exists(datasetpath):
            raise FileNotFoundError("Make sure dataset has been converted!")
        self.X_train, self.y_train, self.X_test, self.y_test = self._ember.read_vectorized_features(datasetpath)
        if self.shuffle:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=32)
        delunknown = (self.y_train != -1)
        self.X_train, self.y_train = self.X_train[delunknown], self.y_train[delunknown]
        delunknown = (self.y_test != -1)
        self.X_test, self.y_test = self.X_test[delunknown], self.y_test[delunknown]
        return self.X_train, self.y_train, self.X_test, self.y_test

    def save_model(self, estimator):
        self._mlflow.sklearn.log_model(estimator, 'skmodel')
        if self.booster == 'lgbm':
            self._mlflow.lightgbm.log_model(estimator.booster_, 'model')
        elif self.booster == 'xgb':
            self._mlflow.lightgbm.log_model(estimator.get_booster(), 'model')
        elif self.booster == 'cb':
            self._mlflow.catboost.log_model(estimator, 'model')

    def train_execution(self):
        self._mlflow.sklearn.autolog()
        if self.booster == 'lgbm':
            self._mlflow.lightgbm.autolog()
            self.model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', n_jobs=self.n_jobs)
        elif self.booster == 'xgb':
            self._mlflow.xgboost.autolog()
            self.model = xgb.XGBClassifier(booster='dart', objective="binary:logistic", n_jobs=self.n_jobs)
        elif self.booster == 'cb':
            self.model = cb.CatBoostClassifier(boosting_type='ordered', thread_count=self.n_jobs)
        self.model.fit(self.X_train, self.y_train)

    def cv_execution(self, n=5, copy=True):
        for cv, (train_ix, test_ix) in enumerate(TimeSeriesSplit(n_splits=n).split(self.X_train)):
            with self._mlflow.start_run(run_name=f'cross_validation_{cv}', nested=True) as child_run:
                cvmodel = deepcopy(self.model) if copy else self.model
                cvmodel.fit(self.X_train[train_ix], self.y_train[train_ix])
                self.y_test_pred = cvmodel.predict(self.X_test)
                self.params(cvmodel, stage=f'{cv}')
                self.metrics(cvmodel, stage=f'{cv}')
                self.save_model(cvmodel)

    def config_params(self, n):
        self.configrun = {
            'session': self.session,
            'experiment': self.experiment,
            'booster': self.booster,
            'cross_validation': n,
            'min_features': self.min_features,
        }
        if features is not None:
            self.configrun['features'] = ','.join(self.features)
        return self.configrun

    def main(self, cv=True, n=3):
        self.startsessiontime = time.time()
        with self._mlflow.start_run(run_name=f'[main]_{self.session}') as run:
            self.config_params(n)
            self.memmetrics = memory_usage(self.train_execution)
            self.y_test_pred = self.model.predict(self.X_test)
            self.params(self.model, stage=f'main')
            self.metrics(self.model, stage=f'main')
            self.save_model(self.model)
            if cv:
                self.cv_execution(n)
            self._mlflow.log_metric('session_time', time.time()-self.startsessiontime)



