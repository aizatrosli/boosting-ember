import joblib, time, os, sys, json, logging, inspect, gc, multiprocessing
from copy import deepcopy
from memory_profiler import memory_usage
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


from ember import *
from features_extended import *
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, make_scorer, average_precision_score

from sklearnex import patch_sklearn
patch_sklearn()


class Boosting(object):

    def __init__(self, session, booster='lgbm', dataset='ember2018', features=None, min_features=20, url=None, configpath='config/fyp.json'):
        import mlflow, ember
        self.startsessiontime, self.memmetrics, self.model = None, None, None
        self._ember = ember
        self._mlflow = mlflow
        self.booster = booster
        self.min_features = min_features

        self.projectpath = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
        self.logger = self.create_logsession(session)
        self.config = self.load_config(os.path.join(self.projectpath, configpath))
        if url is None:
            self.setup_mlflow(os.path.join(self.projectpath, 'config', 'mlflow'))
        else:
            self.setup_mlflow(url)
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(dataset)
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


    def metrics(self, estimator, stage=''):
        self.stage = [stage] if stage else []
        self._mlflow.log_metric(self.keyname('fit_time'), time.time()-self.startsessiontime)
        self._mlflow.sklearn.eval_and_log_metrics(model=estimator, X=self.X_test, y_true=self.y_test, prefix=f'{stage}_')
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
            self._mlflow.log_artifact('memory_training.csv')
            df = df.set_index(['time (seconds)'])
            fig = df.plot.line()
            self._mlflow.log_figure(fig.figure, 'memory_training.png')

    def keyname(self, name):
        return '_'.join(self.stage + [name])

    def load_data(self,datasetpath):
        if not os.path.exists(datasetpath):
            raise FileNotFoundError("Make sure dataset has been converted!")
        return self._ember.read_vectorized_features(datasetpath)

    def save_model(self, estimator):
        self._mlflow.sklearn.log_model(estimator, 'skmodel')
        if self.booster == 'lgbm':
            self._mlflow.lightgbm.log_model(estimator.booster_, 'model')
        elif self.booster == 'xgb':
            self._mlflow.lightgbm.log_model(estimator.get_booster(), 'model')
        elif self.booster == 'cb':
            self._mlflow.catboost.log_model(estimator, 'model')

    def train_execution(self, params):
        self._mlflow.sklearn.autolog()
        if self.booster == 'lgbm':
            self._mlflow.lightgbm.autolog()
            self.model = lgb.LGBMClassifier()
        elif self.booster == 'xgb':
            self._mlflow.xgboost.autolog()
            self.model = xgb.XGBClassifier()
        elif self.booster == 'cb':
            self.model = cb.CatBoostClassifier()
        self.model.fit(self.X_train, self.y_train)

    def cv_execution(self, n=5, copy=True):
        for cv, (train_ix, test_ix) in enumerate(TimeSeriesSplit(n_split=n).split(self.X_train)):
            cvmodel = deepcopy(self.model) if copy else self.model
            cvmodel.fit(self.X_train[train_ix], self.y_train[train_ix])
            self.params(cvmodel, stage=f'{cv}')
            self.metrics(cvmodel, stage=f'{cv}')
            self.save_model(cvmodel)





    def main(self, cv=True, n=3):

        self.startsessiontime = time.time()
        self.memmetrics = memory_usage(self.train_execution, (params,))
        self.save_model(self.model)
        if cv:
            self.cv_execution(n)





def scorecheck():
    '''
    Scoring methodology return values as dictionary
    eg.
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    :return:
    '''
    return {'roc_auc': make_scorer(roc_auc_score, max_fpr=5e-3),
            'precision': make_scorer(average_precision_score),
            'accuracy': make_scorer(accuracy_score)}


def generic_crossvalidation(model, X, y, nsplit=3):
    '''

    explaination:
    Multimetric scoring can either be specified as a list of strings of predefined scores names or a dict mapping the
    scorer name to the scorer function and/or the predefined scorer name(s). See Using multiple metric evaluation for
    more details. When specifying multiple metrics, the refit parameter must be set to the metric (string) for which
    the *best_params_* will be found and used to build the best_estimator_ on the whole dataset. If the search should not
    be refit, set refit=False. Leaving refit to the default value None will result in an error when using multiple metrics.
    :return:
    '''

    cv = TimeSeriesSplit(n_splits=nsplit).split(X)
    return cross_validate(model, X, y, scoring=scorecheck(), cv=cv, n_jobs=multiprocessing.cpu_count()/4, verbose=10)


def mlflowsetup(url, file='mlflow.json'):
    import requests, pickle, os, json
    from urllib.parse import urlparse
    data = pickle.loads(requests.get(url).content) if urlparse(url).scheme else pickle.loads(open(url, 'rb').read())
    for key,val in data.items():
        if type(val) is str:
            os.environ[key]=val
    with open(file, 'w') as fp:
        json.dump(data['gcloud'], fp)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file
    return True


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


def yield_artifacts(run_id, path=None):
    import mlflow
    """Yield all artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    import mlflow
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }


def get_dataset(data_dir, feature_version=2):
    X_train, y_train = read_vectorized_features(data_dir, "train", feature_version)


def read_data_record(raw_features_string):
    """
    Decode a raw features string and return the metadata fields
    """
    all_data = json.loads(raw_features_string)
    return {k: all_data[k] for k in all_data.keys()}


def create_resize_data(data_dir, size=100000, cache=False, random=False):
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()

    train_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    train_metadf = pd.DataFrame(list(pool.imap(read_data_record, raw_feature_iterator(train_feature_paths))))
    train_metadfsize = pd.concat(
        [train_metadf[train_metadf['label'] == 0].head(size), train_metadf[train_metadf['label'] == 1].head(size),
         train_metadf[train_metadf['label'] == -1].head(size)], ignore_index=True) if not random else pd.concat(
        [train_metadf[train_metadf['label'] == 0].sample(size), train_metadf[train_metadf['label'] == 1].sample(size),
         train_metadf[train_metadf['label'] == -1].sample(size)], ignore_index=True)
    logging.info(f'Train original size : {train_metadf.shape} | Train resize size : {train_metadfsize.shape}')
    if cache:
        train_metadf.to_pickle(os.path.join(data_dir, "train.data"), compression=None)
    del train_metadf
    train_metadfsize.to_pickle(os.path.join(data_dir, f"train_{size * 3}.data"), compression=None)
    del train_metadfsize
    gc.collect()

    test_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    test_metadf = pd.DataFrame(list(pool.imap(read_data_record, raw_feature_iterator(test_feature_paths))))
    test_metadf.to_pickle(os.path.join(data_dir, "test.data"), compression=None)
    logging.info(f'Test size : {test_metadf.shape}')
    del test_metadf
    gc.collect()

    return data_dir


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
