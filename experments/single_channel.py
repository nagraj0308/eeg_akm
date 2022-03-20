import pandas as pd

from common.tools import get_tms_tmr, get_diagonal_averaging
from config import OUTPUT_PATH
from techniques.kmean_clustering import k_mean_clustering


def single_channel(COL):
    path_mat = OUTPUT_PATH + 'mat.csv'
    path_amc = OUTPUT_PATH + 'amc.csv'
    data_mat = pd.read_csv(path_mat)
    data_amc = pd.read_csv(path_amc)
    result = k_mean_clustering(data_amc)
    tms, tmr = get_tms_tmr(data_mat, result)
    true_eeg = get_diagonal_averaging(tms)
    noise = get_diagonal_averaging(tmr)

    # output
    f = pd.DataFrame(true_eeg)
    f.to_csv(OUTPUT_PATH + "filtered.csv", index=False)
    f = pd.DataFrame(noise)
    f.to_csv(OUTPUT_PATH + "noise.csv", index=False)
