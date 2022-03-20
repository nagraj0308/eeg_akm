import math

import numpy as np
import pandas as pd

from common.graphs import all_graphs
from config import SIZE, PATH, COLS, ROWS, OUTPUT_PATH


def create_matrix_file(COL):
    file = pd.read_csv(PATH)  # (37481, 15)
    data = file.iloc[0:SIZE, COL]  # (37481,) channel 0 data (single channel data)
    mat = pd.DataFrame()
    for i in range(COLS):
        tmp = data.iloc[i:(i + ROWS)].to_numpy()
        mat = pd.concat([mat, pd.Series(tmp)], axis=1)
    mat.to_csv('./data/output/mat' + ".csv", index=False)

    amc = []
    for i in range(COLS):
        vector = mat.iloc[:, i]
        activity = np.var(np.array(vector))
        mc = hjorth(vector)
        amc.append([activity, mc[0], mc[1]])
    f = pd.DataFrame(amc)
    f.to_csv("./data/output/amc" + ".csv", index=False)


def hjorth(X, D=None):
    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return np.sqrt(M2 / TP), np.sqrt(
        float(M4) * TP / M2 / M2
    )  # Hjorth Mobility and Complexity


def get_tms_tmr(matrix, result):
    tms = matrix.copy()
    tmr = matrix.copy()
    mi = np.bincount(result).argmax()
    n = np.size(result)
    k = math.ceil(ROWS / 2)
    for i in range(0, k):
        tmr.iloc[:, i] = 0
    for i in range(k, n - k):
        if result[i] == mi:
            tmr.iloc[:, i] = 0
        else:
            tms.iloc[:, i] = 0
    for i in range(n - k, n):
        tmr.iloc[:, i] = 0
    return tms, tmr


def get_diagonal_averaging(mat):
    N = ROWS
    K = COLS
    M = SIZE
    d = []
    for n in range(1, M + 1):
        if 1 <= n <= N:
            s = 0
            for i in range(1, n + 1):
                s += mat.iloc[i - 1, (n - 1) - i + 1]
            d.append(s / n)
        elif N < n <= K:
            s = 0
            for i in range(1, N + 1):
                s += mat.iloc[i - 1, (n - 1) - i + 1]
            d.append(s / N)
        elif K < n <= M:
            s = 0
            for i in range((n - K + 1), (M - K + 1) + 1):
                s += mat.iloc[i - 1, (n - 1) - i + 1]
            d.append(s / (M - n + 1))
        else:
            d.append(0)
    return np.array(d)


def show_results(COL):
    file = pd.read_csv(PATH)
    data = file.iloc[0:SIZE, COL]
    filtered_path = OUTPUT_PATH + 'filtered.csv'
    noise_path = OUTPUT_PATH + 'noise.csv'
    filtered_data = pd.read_csv(filtered_path)
    noise = pd.read_csv(noise_path)
    all_graphs(np.array(data), np.array(filtered_data), np.array(noise))
