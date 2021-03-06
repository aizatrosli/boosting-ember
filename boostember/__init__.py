import joblib, time, os, sys, json, logging, inspect, gc, multiprocessing
from copy import deepcopy
from memory_profiler import memory_usage
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

from ember import *
from .debug import *
from .features_extended import *
from .utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from sklearnex import patch_sklearn
#patch_sklearn()


class Boosting(object):

    def __init__(self, session, experiment='Demo', booster='lgb', dataset='ember2018', defaultdataset=True,
                 features=None, shuffle=False, n_estimator=100, n_jobs=0, min_features=20, url=None,
                 configpath='config/fyp.json', verbose=True, notime=True):
        import mlflow, ember
        self.boostercol = {'cb': 'catbooster', 'lgb': 'lightgbm', 'xgb': 'xgboost'}
        self.startsessiontime, self.memmetrics, self.model = None, None, None
        self.shuffle = shuffle
        self.session = session
        self.notime = notime
        self.experiment = experiment
        self._ember = ember
        self.verbose = verbose
        self._mlflow = mlflow
        self.booster = booster
        self.min_features = min_features
        self.n_jobs = int(n_jobs)
        self.n_estimator = int(n_estimator)
        self.defaultdata = defaultdataset
        self.max_fpr = [0.01, 0.001]
        self.projectpath = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
        self.logger = self.create_logsession(session)
        self.config = self.load_config(os.path.join(self.projectpath, configpath))
        if url is None:
            self.setup_mlflow(os.path.join(self.projectpath, 'config', 'mlflow'))
        else:
            self.setup_mlflow(url)
        if not verbose:
            warnings.filterwarnings('ignore')
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(dataset, default=self.defaultdata)
        self.y_train_pred, self.y_test_pred = None, None
        self.features = emberfeatures().features if self.defaultdata else self.X_train.columns.tolist()
        if self.X_train.shape[1] != len(self.features):
            raise("Train features are not match with reference features list!!!")
        if features is not None and isinstance(features, list):
            if isinstance(self.X_train, pd.DataFrame):
                features_ix = features.copy()
                self.X_train, self.X_test = self.X_train[features_ix], self.X_test[features_ix]
            else:
                features_ix = [self.features.index(x) for i, x in enumerate(features)]
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

    def params(self, estimator, run, stage=''):
        self.stage = [stage] if stage else []
        params = estimator.get_all_params() if self.booster == 'cb' else estimator.get_params()
        self._mlflow.log_params(params)
        #self._mlflow.log_params(self.configrun)
        with open('features_list.txt', 'w') as fp:
            fp.write('\n'.join(self.features))
        self._mlflow.log_artifact('features_list.txt')

    def metrics(self, estimator, run, stage=''):
        self.stage = [stage] if stage else []
        self._mlflow.log_metric(self.keyname('fit_time'), int((time.time() - self.startsessiontime) % 60))
        self._mlflow.log_metric(self.keyname('fit_time_raw'), int(time.time() - self.startsessiontime))
        #self._mlflow.sklearn.eval_and_log_metrics(model=estimator, X=self.X_test, y_true=self.y_test, prefix=f'{stage}_')
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
        df = pd.DataFrame({'y_true': self.y_test, 'y_pred': self.y_test_pred})
        df.to_csv(f'testing_dataset_{stage}.csv', index=False)
        self._mlflow.log_artifact(f'testing_dataset_{stage}.csv')
        if self.booster == 'cb':
            self._mlflow.log_artifacts('catboost_info', 'catboost_info')
            self._mlflow.sklearn.utils._log_estimator_content(estimator=estimator, X=self.X_test, y_true=self.y_test,
                                                              prefix=f'{stage}_', run_id=run.info.run_id, sample_weight=None)
        else:
            self._mlflow.sklearn.eval_and_log_metrics(model=estimator, X=self.X_test, y_true=self.y_test, prefix=f'{stage}_')
        plt.close('all')

    def keyname(self, name):
        return '_'.join(self.stage + [name])

    def load_data(self, datasetpath, default=False):
        if not os.path.exists(datasetpath):
            raise FileNotFoundError("Make sure dataset has been converted!")
        if default:
            self.X_train, self.y_train, self.X_test, self.y_test = self._ember.read_vectorized_features(datasetpath)
            if self.shuffle:
                self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=32)
            delunknown = (self.y_train != -1)
            self.X_train, self.y_train = self.X_train[delunknown], self.y_train[delunknown]
            delunknown = (self.y_test != -1)
            self.X_test, self.y_test = self.X_test[delunknown], self.y_test[delunknown]

        else:
            traindf = pd.read_pickle(os.path.join(datasetpath, 'ember2018_ft_train.data'))
            testdf = pd.read_pickle(os.path.join(datasetpath, 'ember2018_ft_test.data'))
            dropcol = [col for col in traindf.columns.tolist() if 'timestamp' in col]
            dropcol = ['label']+dropcol if self.notime else 'label'
            self.X_train, self.y_train = traindf.drop(dropcol, axis=1), traindf['label'].astype('float32')
            self.X_test, self.y_test = testdf.drop(dropcol, axis=1), testdf['label'].astype('float32')
        if self.verbose:
            print(f'X_train: {self.X_train.shape}\n y_train: {self.y_train.shape}\nX_test: {self.X_test.shape}\n y_test: {self.y_test.shape}\n')
        return self.X_train, self.y_train, self.X_test, self.y_test

    def save_model(self, estimator):
        self._mlflow.sklearn.log_model(estimator, 'skmodel')
        if self.booster == 'lgb':
            self._mlflow.lightgbm.log_model(estimator.booster_, 'model')
        elif self.booster == 'xgb':
            self._mlflow.lightgbm.log_model(estimator.get_booster(), 'model')
        elif self.booster == 'cb':
            self._mlflow.catboost.log_model(estimator, 'model')

    def train_execution(self):
        verbose = 5 if self.verbose else -1
        silent = False if self.verbose else True
        self._mlflow.sklearn.autolog(silent=silent)
        if self.booster == 'lgb':
            self._mlflow.lightgbm.autolog(silent=silent)
            self.model = lgb.LGBMClassifier(boosting_type='goss', objective='binary', n_estimators=self.n_estimator, n_jobs=self.n_jobs, verbose=verbose)
        elif self.booster == 'xgb':
            self._mlflow.xgboost.autolog(silent=silent)
            self.model = xgb.XGBClassifier(booster='dart', objective="binary:logistic", n_estimators=self.n_estimator, n_jobs=self.n_jobs, silent=silent)
        elif self.booster == 'cb':
            self.model = cb.CatBoostClassifier(boosting_type='Ordered', n_estimators=self.n_estimator, thread_count=self.n_jobs, verbose=verbose)
        self.model.fit(self.X_train, self.y_train, verbose=self.verbose)

    def cv_execution(self, n=5, copy=True):
        for cv, (train_ix, test_ix) in enumerate(TimeSeriesSplit(n_splits=n).split(self.X_train)):
            with self._mlflow.start_run(run_name=f'cross_validation_{cv}', tags=self.config_params(cv), nested=True) as child_run:
                cvmodel = deepcopy(self.model) if copy else self.model
                if self.verbose:
                    print(train_ix, test_ix)
                if isinstance(self.X_train, pd.DataFrame):
                    cvmodel.fit(self.X_train.iloc[train_ix], self.y_train.iloc[train_ix], verbose=self.verbose)
                else:
                    cvmodel.fit(self.X_train[train_ix], self.y_train[train_ix], verbose=self.verbose)
                self.y_test_pred = cvmodel.predict(self.X_test)
                self.params(cvmodel, child_run, stage=f'{cv}')
                self.metrics(cvmodel, child_run, stage=f'{cv}')
                self.save_model(cvmodel)

    def config_params(self, n):
        self.configrun = {
            'defaultdataset': str(self.defaultdata),
            'n_features': str(self.X_train.shape[1]),
            'boosting_model': self.boostercol[self.booster],
            'session': self.session,
            'experiment': self.experiment,
            'cross_validation': str(n),
            'min_features': str(self.min_features),
        }
        return self.configrun

    def main(self, cv=True, n=3):
        self.startsessiontime = time.time()
        with self._mlflow.start_run(run_name=f'main', tags=self.config_params(n)) as run:
            self.config_params(n)
            self.memmetrics = memory_usage(self.train_execution)
            self.y_test_pred = self.model.predict(self.X_test)
            self.params(self.model, run, stage=f'main')
            self.metrics(self.model, run, stage=f'main')
            self.save_model(self.model)
            if cv:
                self.cv_execution(n)
            self._mlflow.log_metric('session_time', int((time.time() - self.startsessiontime) % 60))



