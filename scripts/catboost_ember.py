from boostember import *

run = Boosting('testing catboost', booster='cb',
               dataset='/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018', n_jobs=22)
run.main(cv=False)