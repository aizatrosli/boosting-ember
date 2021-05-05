import os,sys

sys.path.append(os.path.dirname(os.getcwd()))
from boostember import *

run = Boosting('testing xgboost', booster='xgb', dataset='/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018', n_jobs=22)
run.main(cv=True, n=10)