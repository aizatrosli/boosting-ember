import os,sys,time,json,tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
import logging, inspect
from ember.features import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, make_scorer)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logfile.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

'''
logger.debug('A debug message')
logger.info('An info message')
logger.warning('Something is not right.')
logger.error('A Major error has happened.')
logger.critical('Fatal error. Cannot continue')
'''


def autolog(message):
    "Automatically log the current function details."
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    func = inspect.currentframe().f_back.f_code
    # Dump the message + the name of this function to the log.
    logger.debug("%s: %s in %s:%i" % (
        message,
        func.co_name,
        func.co_filename,
        func.co_firstlineno
    ))


def raw_feature_iterator(file_paths):
    """
    Yield raw feature strings from the inputed file paths
    """
    i = 0
    for path in file_paths:
        with open(path, "r") as fin:
            for line in fin:
                i += 1
                autolog('No.{} "{}" {}'.format(i, line, path))
                yield line


def vectorize(irow, raw_features_string, X_path, y_path, extractor, nrows):
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)
    autolog('{}_{}')

    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
    y[irow] = raw_features["label"]

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows):
    """
    Vectorize a subset of data and write it to disk
    """
    # Create space on disk to write features to
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = ((irow, raw_features_string, X_path, y_path, extractor, nrows)
                         for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)))
    autolog('{}\n{}'.format(type(argument_iterator), argument_iterator))
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        autolog(_)
        pass


def create_vectorized_features(data_dir, feature_version=2):
    """
    Create feature vectors from raw features and write them to disk
    """
    extractor = PEFeatureExtractor(feature_version)

    print("Vectorizing training set")
    X_path = os.path.join(data_dir, "X_train.dat")
    y_path = os.path.join(data_dir, "y_train.dat")
    raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
    vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows)

    print("Vectorizing test set")
    X_path = os.path.join(data_dir, "X_test.dat")
    y_path = os.path.join(data_dir, "y_test.dat")
    raw_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
    vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows)


def read_vectorized_features(data_dir, subset=None, feature_version=2):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    if subset is not None and subset not in ["train", "test"]:
        return None

    extractor = PEFeatureExtractor(feature_version)
    ndim = extractor.dim
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if subset is None or subset == "train":
        X_train_path = os.path.join(data_dir, "X_train.dat")
        y_train_path = os.path.join(data_dir, "y_train.dat")
        y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
        N = y_train.shape[0]
        X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "train":
            return X_train, y_train

    if subset is None or subset == "test":
        X_test_path = os.path.join(data_dir, "X_test.dat")
        y_test_path = os.path.join(data_dir, "y_test.dat")
        y_test = np.memmap(y_test_path, dtype=np.float32, mode="r")
        N = y_test.shape[0]
        X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "test":
            return X_test, y_test

    return X_train, y_train, X_test, y_test


def read_metadata_record(raw_features_string):
    """
    Decode a raw features string and return the metadata fields
    """
    all_data = json.loads(raw_features_string)
    metadata_keys = {"sha256", "appeared", "label", "avclass"}
    return {k: all_data[k] for k in all_data.keys() & metadata_keys}


def create_metadata(data_dir):
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()

    train_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    train_records = list(pool.imap(read_metadata_record, raw_feature_iterator(train_feature_paths)))

    metadata_keys = ["sha256", "appeared", "label", "avclass"]
    ordered_metadata_keys = [k for k in metadata_keys if k in train_records[0].keys()]

    train_metadf = pd.DataFrame(train_records)[ordered_metadata_keys]
    train_metadf.to_csv(os.path.join(data_dir, "train_metadata.csv"))

    train_records = [dict(record, **{"subset": "train"}) for record in train_records]

    test_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    test_records = list(pool.imap(read_metadata_record, raw_feature_iterator(test_feature_paths)))

    test_metadf = pd.DataFrame(test_records)[ordered_metadata_keys]
    test_metadf.to_csv(os.path.join(data_dir, "test_metadata.csv"))

    test_records = [dict(record, **{"subset": "test"}) for record in test_records]

    all_metadata_keys = ordered_metadata_keys + ["subset"]
    metadf = pd.DataFrame(train_records + test_records)[all_metadata_keys]
    metadf.to_csv(os.path.join(data_dir, "metadata.csv"))
    return metadf


def read_metadata(data_dir):
    """
    Read an already created metadata file and return its dataframe
    """
    return pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)


def optimize_model(data_dir):
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
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'num_iterations': [500, 1000],
        'learning_rate': [0.005, 0.05],
        'num_leaves': [512, 1024, 2048],
        'feature_fraction': [0.5, 0.8, 1.0],
        'bagging_fraction': [0.5, 0.8, 1.0]
    }
    model = lgb.LGBMClassifier(boosting_type="gbdt", n_jobs=-1, silent=True)

    # each row in X_train appears in chronological order of "appeared"
    # so this works for progrssive time series splitting
    progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train)

    grid = GridSearchCV(estimator=model, cv=progressive_cv, param_grid=param_grid, scoring=score, n_jobs=1, verbose=3)
    grid.fit(X_train, y_train)

    return grid.best_params_


def train_model(data_dir, params={}, feature_version=2):
    """
    Train the LightGBM model from the EMBER dataset from the vectorized features
    """
    # update params
    params.update({"application": "binary"})

    # Read data
    X_train, y_train = read_vectorized_features(data_dir, "train", feature_version)

    # Filter unlabeled data
    train_rows = (y_train != -1)

    # Train
    lgbm_dataset = lgb.Dataset(X_train[train_rows], y_train[train_rows])
    lgbm_model = lgb.train(params, lgbm_dataset)

    return lgbm_model


def predict_sample(lgbm_model, file_data, feature_version=2):
    """
    Predict a PE file with an LightGBM model
    """
    extractor = PEFeatureExtractor(feature_version)
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return lgbm_model.predict([features])[0]