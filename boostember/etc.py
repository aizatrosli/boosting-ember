import multiprocessing
import json,os
import pandas as pd
import logging
from ember import *

data_dir2018 = '/root/dataset/ember2018/'
data_dir2017 = '/root/dataset/ember_2017_2/'





create_fdata(data_dir2018)
create_fdata(data_dir2017)