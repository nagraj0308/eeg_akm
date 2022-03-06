import numpy as np
import pandas as pd

from config import SIZE, PATH, COL, COLS, ROWS


def create_matrix_file():
    file = pd.read_csv(PATH)  # (37481, 15)
    data = file.iloc[0:SIZE, COL]  # (37481,) channel 0 data (single channel data)
    print("------MAT------")
    mat = pd.DataFrame()
    for i in range(COLS):
        print('matrix', i)
        tmp = data.iloc[i:(i + ROWS)].to_numpy()
        mat = pd.concat([mat, pd.Series(tmp)], axis=1)
    mat.to_csv('./data/output/mat' + ".csv", index=False)

    print("------ACM------")
    amc = []
    for i in range(COLS):
        print('amc', i)
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
    for i in range(n):
        if result[i] == mi:
            tmr.iloc[:, i] = 0
        else:
            tms.iloc[:, i] = 0
    return tms, tmr


def get_diagonal_averaging(mat):
    N = ROWS
    K = COLS
    M = SIZE
    d = []
    for n in range(1, M + 1):
        print(n)
        if 1 >= n and n < N:
            s = 0
            for i in range(0, n):
                s += mat.iloc[i, (n - 1) - i + 1]
            d.append(s / n)
        elif N >= n and n <= K:
            s = 0
            for i in range(0, N):
                s += mat.iloc[i, (n - 1) - i + 1]
            d.append(s / N)
        elif K < n and n <= M:
            s = 0
            for i in range((n - K + 1), (M - K + 1)):
                s += mat.iloc[i, (n - 1) - i + 1]
            d.append(s / (M - n + 1))
        else:
            d.append(0)
    return np.array(d)


def diagonal_avg(data):
    a = data
    diags = [a[::-1, :].diagonal(i) for i in range(-a.shape[0] + 1, a.shape[1])]
    # Now back to the original array to get the upper-left-to-lower-right diagonals,
    # starting from the right, so the range needed for shape (x,y) was y-1 to -x+1 descending.
    diags.extend(a.diagonal(i) for i in range(a.shape[1] - 1, -a.shape[0], -1))
    # Another list comp to convert back to Python lists from numpy arrays, # so it prints what you requested.
    print([n.tolist() for n in diags])

# x= np.arange(0,16).reshape(4,4)
# diagonal_avg(x)
