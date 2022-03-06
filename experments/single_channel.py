import numpy as np
import pandas as pd

from common.graphs import all_graphs
from common.tools import get_tms_tmr, get_diagonal_averaging
from config import PATH, SIZE, COL
from techniques.kmean_clustering import k_mean_clustering


def single_channel():
    path_mat = './data/output/mat.csv'
    path_amc = './data/output/amc.csv'
    data_mat = pd.read_csv(path_mat)
    data_amc = pd.read_csv(path_amc)

    result = k_mean_clustering(data_amc)
    print(result)

    tms, tmr = get_tms_tmr(data_mat, result)
    print(tms)
    print(tmr)
    true_eeg = get_diagonal_averaging(tms)
    noise = get_diagonal_averaging(tmr)

    print(true_eeg)
    print(noise)


    # output
    file = pd.read_csv(PATH)
    data = file.iloc[0:SIZE, COL]
    all_graphs(np.array(data), np.array(true_eeg), np.array(noise))
