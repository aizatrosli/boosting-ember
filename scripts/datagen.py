import pandas as pd
import multiprocessing
import json, os, joblib
from ember import *


def read_data_record(raw_features_string):
    all_data = json.loads(raw_features_string)
    return {k: all_data[k] for k in all_data.keys()}


def generatecollection(xdf):
    ftdf = {}
    df = xdf.copy()

    ndf = df[['sha256', 'appeared', 'label']]
    ftdf['metadata'] = ndf.copy()

    ndf = df[['sha256', 'histogram']]
    ndf = ndf.drop('histogram', axis=1).join(
        pd.DataFrame(ndf['histogram'].dropna().tolist(), index=ndf['histogram'].dropna().index).add_prefix(
            'histogram_'))
    ndf = ndf.reset_index().rename(columns={'index': 'histogram_id'})
    ftdf['histogram'] = ndf.copy()

    ndf = df[['sha256', 'byteentropy']]
    ndf = ndf.drop('byteentropy', axis=1).join(
        pd.DataFrame(ndf['byteentropy'].dropna().tolist(), index=ndf['byteentropy'].dropna().index).add_prefix(
            'byteentropy_'))
    ndf = ndf.reset_index().rename(columns={'index': 'byteentropy_id'})
    ftdf['byteentropy'] = ndf.copy()

    ndf = df[['sha256', 'strings']]
    ndf = ndf.drop('strings', axis=1).join(pd.DataFrame(df['strings'].values.tolist()))
    ndf = ndf.drop('printabledist', axis=1).join(
        pd.DataFrame(ndf['printabledist'].dropna().tolist(), index=ndf['printabledist'].dropna().index).add_prefix(
            'printabledist_'))
    ndf = ndf.reset_index().rename(columns={'index': 'strings_id'})
    ftdf['strings'] = ndf.copy()

    ndf = df[['sha256', 'general']]
    ndf = ndf.drop('general', axis=1).join(pd.DataFrame(df['general'].values.tolist()))
    ndf = ndf.reset_index().rename(columns={'index': 'general_id'})
    ftdf['general'] = ndf.copy()

    ndf = df[['sha256', 'header']]
    ndf = ndf.drop('header', axis=1).join(pd.DataFrame(df['header'].values.tolist(), index=df.index))
    mdf = ndf.drop(['coff', 'optional'], axis=1).join(
        pd.DataFrame(ndf['coff'].values.tolist(), index=ndf.index).add_prefix('coff_'))
    mdf = mdf.reset_index().rename(columns={'index': 'header_coff_id'})
    ftdf['header_coff'] = mdf.drop('coff_characteristics', axis=1).copy()

    odf = mdf[['sha256', 'coff_characteristics']]
    odf['coff_characteristics_id'] = [[j for j in range(len(i))] for i in odf['coff_characteristics'].values]
    odf.set_index(['sha256'])
    odf = odf.apply(pd.Series.explode).dropna().reset_index().drop('index', axis=1)
    odf['coff_characteristics_id'] = odf['sha256'] + "_" + odf['coff_characteristics_id'].astype(str)
    ftdf['coff_characteristics'] = odf.copy()

    mdf = ndf.drop(['coff', 'optional'], axis=1).join(
        pd.DataFrame(ndf['optional'].values.tolist(), index=ndf.index).add_prefix('optional_'))
    mdf = mdf.reset_index().rename(columns={'index': 'header_optional_id'})
    ftdf['header_optional'] = mdf.drop('optional_dll_characteristics', axis=1).copy()

    odf = mdf[['sha256', 'optional_dll_characteristics']]
    odf['optional_dll_characteristics_id'] = [[j for j in range(len(i))] for i in
                                              odf['optional_dll_characteristics'].values]
    odf.set_index(['sha256'])
    odf = odf.apply(pd.Series.explode).dropna().reset_index().drop('index', axis=1)
    odf['optional_dll_characteristics_id'] = odf['sha256'] + "_" + odf['optional_dll_characteristics_id'].astype(str)
    ftdf['optional_dll_characteristics'] = odf.copy()

    ndf = df[['sha256', 'section']]
    ndf = ndf.drop('section', axis=1).join(pd.DataFrame(df['section'].values.tolist(), index=df.index))
    ndf = ndf.reset_index().rename(columns={'index': 'section_id'})
    ftdf['section'] = ndf.drop('sections', axis=1).copy()

    odf = ndf[['sha256', 'sections']]
    odf['sections_id'] = [[j for j in range(len(i))] for i in odf['sections'].values]
    odf.set_index(['sha256'])
    odf = odf.apply(pd.Series.explode).dropna().reset_index().drop('index', axis=1)
    odf = odf.drop('sections', axis=1).join(pd.DataFrame(odf['sections'].values.tolist()))
    odf['sections_id'] = odf['sha256'] + "_" + odf['sections_id'].astype(str)
    ftdf['section_sections'] = odf.drop('props', axis=1).copy()

    pdf = odf[['sha256', 'sections_id', 'props']].dropna()
    pdf['props_id'] = [[j for j in range(len(i))] for i in pdf['props'].values]
    pdf.set_index(['sha256', 'sections_id'])
    pdf = pdf.apply(pd.Series.explode).dropna().reset_index().drop('index', axis=1)
    pdf['props_id'] = pdf['sections_id'] + "_" + pdf['props_id'].astype(str)
    ftdf['sections_props'] = pdf.copy()

    ndf = df[['sha256', 'imports']]
    ndf.set_index(['sha256'])
    ndf['imports_api'] = [i.values() for i in ndf['imports'].values]
    ndf['imports_id'] = [[j for j in range(len(i.keys()))] for i in ndf['imports'].values]
    ndf['imports'] = [i.keys() for i in ndf['imports'].values]
    ndf = ndf.apply(pd.Series.explode).reset_index().drop('index', axis=1)
    ndf['imports_id'] = ndf['sha256'] + "_" + ndf['imports_id'].astype(str)
    ftdf['imports'] = ndf.drop('imports_api', axis=1).copy()

    odf = ndf[['sha256', 'imports_id', 'imports_api']].dropna()
    odf.set_index(['sha256', 'imports_id'])
    odf['imports_api_id'] = [[j for j in range(len(i))] for i in odf['imports_api'].values]
    odf = odf.apply(pd.Series.explode).dropna().reset_index().drop('index', axis=1)
    odf['imports_api_id'] = odf['imports_id'] + "_" + odf['imports_api_id'].astype(str)
    ftdf['imports_api'] = odf.copy()

    ndf = df[['sha256', 'exports']]
    ndf.set_index(['sha256'])
    ndf['exports_id'] = [[j for j in range(len(i))] for i in ndf['exports'].values]
    ndf = ndf.apply(pd.Series.explode).dropna().reset_index().drop('index', axis=1)
    ndf['exports_id'] = ndf['sha256'] + "_" + ndf['exports_id'].astype(str)
    ftdf['exports'] = ndf.copy()

    ndf = df[['sha256', 'datadirectories']]
    ndf.set_index(['sha256'])
    ndf['datadirectories_id'] = [[j for j in range(len(i))] for i in ndf['datadirectories'].values]
    ndf = ndf.apply(pd.Series.explode).dropna().reset_index().drop('index', axis=1)
    ndf = ndf.drop('datadirectories', axis=1).join(
        pd.DataFrame(ndf['datadirectories'].values.tolist(), index=ndf.index))
    ndf['datadirectories_id'] = ndf['sha256'] + "_" + ndf['datadirectories_id'].astype(str)
    ftdf['datadirectories'] = ndf.copy()
    return ftdf


def convertdf(data_dir):
    pool = multiprocessing.Pool()
    #train_feature_paths = [os.path.join(data_dir, f'train_features_{i}.jsonl') for i in range(6)]
    train_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    train_metadf = pd.DataFrame(list(pool.imap(read_data_record, raw_feature_iterator(train_feature_paths))))
    train_metadf = train_metadf[train_metadf['label'] != -1]
    # train_metadf = train_metadf.sample(100000)
    return generatecollection(train_metadf)


if __name__ == '__main__':
    ember2018 = r'/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018'
    joblib.dump(convertdf(ember2018), os.path.join(ember2018, 'ember2018_ft_test.data'))

