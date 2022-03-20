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


def all_graphs(y1, y2, y3):
    fig, axs = plt.subplots(3, figsize=(24, 8))

    g1 = axs[0]
    g2 = axs[1]
    g3 = axs[2]

    g1.set_title('Eye blink contaminated EEG signal')
    g1.plot(y1, color='blue', label='Eye blink contaminated EEG signal')
    g1.set_ylim([-40, 40])

    g2.set_title('Extracted EEG component')
    g2.plot(y2, color='blue', label='Extracted EEG component')
    g2.set_ylim([-40, 40])

    g3.set_title('Extracted eye blink component')
    g3.plot(y3, color='blue', label='Extracted eye blink component')
    g3.set_ylim([-40, 40])

    fig.subplots_adjust(hspace=0.8)
    plt.show()
