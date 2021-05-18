import math,mlflow

config = {}


def setup_mlflow(url, file='mlflow.json'):
    import requests, pickle
    from urllib.parse import urlparse
    data = pickle.loads(requests.get(url).content) if urlparse(url).scheme else pickle.loads(open(url, 'rb').read())
    config.update(data)
    for key, val in data.items():
        if isinstance(val, str):
            os.environ[key] = val
    with open(file, 'w') as fp:
        json.dump(data['gcloud'], fp)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file
    mlflow.set_tracking_uri(config['tracking_uri'])


def getdata(row):
    for key,val in row.items():
        if val is not None and 'metric' in key:
            for col in targetcol:
                if col in key and not math.isnan(val):
                    row['final_metric.'+col] = val
    return row


def run(url, filterstr='tags.session LIKE "%n100 %"'):
    config['tracking_uri'] = "https://atlascompanion.live/"
    setup_mlflow(url, file='mlflow.json')
    df = mlflow.search_runs(experiment_ids=[5], filter_string=filterstr)
    df = df.apply(getdata, axis=1)
    return df

