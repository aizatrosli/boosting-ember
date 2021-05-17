import os,sys,gc,itertools,tqdm

sys.path.append(os.path.dirname(os.getcwd()))
from boostember import *

config = {
    'booster': ['xgb'],
    'experiment': ['emberboosting'],
    'n_estimator': [10, 50, 100, 500, 1000],
    'defaultdataset': [True],
}


def run():

    keys, values = zip(*config.items())
    experiments = tqdm.tqdm([dict(zip(keys, v)) for v in itertools.product(*values)])

    for experimentdict in experiments:
        exp = 'no AFE without featurehaser' if experimentdict['defaultdataset'] else 'AFE without featurehaser'
        experiments.set_description("{} \n".format("\t".join(f"[{k}]: {v}" for k, v in experimentdict.items())))
        run = Boosting(f'{experimentdict["booster"]} n{experimentdict["n_estimator"]} {exp}',
                       experiment=experimentdict['experiment'], booster=experimentdict["booster"],
                       n_estimator=experimentdict["n_estimator"], defaultdataset=experimentdict["defaultdataset"],
                       dataset='/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018', n_jobs=21, verbose=False)
        run.main(cv=True, n=5)
        del run
        gc.collect()


if __name__ == '__main__':
    run()

