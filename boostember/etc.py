import multiprocessing
import json,os
import pandas as pd
from ember import *

data_dir2018 = '/root/dataset/ember2018/'
data_dir2017 = '/root/dataset/ember_2017_2/'


def read_data_record(raw_features_string):
    """
    Decode a raw features string and return the metadata fields
    """
    all_data = json.loads(raw_features_string)
    return {k: all_data[k] for k in all_data.keys()}


def create_fdata(data_dir):
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()

    train_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    train_metadf = pd.DataFrame(list(pool.imap(read_data_record, raw_feature_iterator(train_feature_paths))))
    train_metadf300k = pd.concat([train_metadf[train_metadf['label'] == 0].head(100000), train_metadf[train_metadf['label'] == 1].head(100000), train_metadf[train_metadf['label'] == -1].head(100000)],ignore_index=True)
    train_metadf.to_pickle(os.path.join(data_dir, "train.data"), compression=None)
    train_metadf300k.to_pickle(os.path.join(data_dir, "train_300k.data"), compression=None)
    trainsize = str(train_metadf.shape)
    print(trainsize, train_metadf.head())
    del train_metadf
    del train_metadf300k

    test_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    test_metadf = pd.DataFrame(list(pool.imap(read_data_record, raw_feature_iterator(test_feature_paths))))
    test_metadf.to_pickle(os.path.join(data_dir, "test.data"), compression=None)
    testsize = str(test_metadf.shape)
    print(testsize)
    del test_metadf

    return trainsize, testsize


create_fdata(data_dir2018)
create_fdata(data_dir2017)