import os,sys

sys.path.append(os.path.dirname(os.getcwd()))
from boostember import *

run = Boosting('testing lightgbm', booster='lgb', dataset='/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018', n_jobs=20)
run.main(cv=True, n=5)
