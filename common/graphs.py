import matplotlib.pyplot as plt
import numpy as np


def plot_scatters(data, title, label_x, label_y):
    y = np.array(data)
    n = np.size(data)
    x = range(0, n)
    plt.scatter(x, y, s=2, marker='.', c='r')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()


def plot_wave(data, title, label_x, label_y):
    y = np.array(data)
    n = np.size(data)
    x = range(0, n)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(x, y, lw=1.5, color='k')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xlim([x[0], x[n - 1]])
    plt.title(title)
    plt.show()


def all_graphs(y1, y2, y3):
    fig, axs = plt.subplots(3, figsize=(24, 8))

    g1 = axs[0]
    g2 = axs[1]
    g3 = axs[2]

    g1.set_title('All Data')
    g1.plot(y1, color='blue', label='All Data')
    g1.set_ylim([-40, 40])

    g2.set_title('Filtered Data')
    g2.plot(y2, color='blue', label='Filtered Data')
    g2.set_ylim([-40, 40])

    g3.set_title('Noise Data')
    g3.plot(y3, color='blue', label='Noise Data')
    g3.set_ylim([-40, 40])

    fig.subplots_adjust(hspace=0.8)
    plt.show()