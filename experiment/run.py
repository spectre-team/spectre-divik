"""Run DiviK for given dataset

Arguments:
    source of tabular data
    destination of result file

"""
from multiprocessing import Pool
import sys

import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm_notebook

import spdivik.predefined


if __name__ == '__main__':
    data = np.load(sys.argv[1])
    with Pool() as pool, tqdm_notebook(desc='processed', total=data.shape[0]) as progress:
        result = spdivik.predefined.master(pool=pool, progress_reporter=progress)(data)
    joblib.dump(result, sys.argv[2])
