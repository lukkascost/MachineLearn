import time

from cv2 import *
import cv2.ml as ml
from matplotlib import cm

from Classes import *
import numpy as np

import matplotlib.pyplot as plt

from Classes.Cluster.SOM import SOM, distance

COLOR = cm.rainbow(np.linspace(0, 1, 8))

n_neurons = 40
i_learning_rate = 0.01
epochs = 2
n_class = 7

k_nn = 5

use_train = 0.8

att = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=[x for x in range(24)], delimiter=",")
label = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=-1, dtype=object, delimiter=",")

# initial neighbourhood radius
init_radius = np.max([n_neurons, n_neurons]) / 2
# radius decay parameter

oDataSet = DataSet()
for j, i in enumerate(att):
    oDataSet.add_sample_of_attribute(np.array(list(i) + [label[j]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
# oDataSet.normalize_data_set()
# oDataSet.attributes = np.nan_to_num((oDataSet.attributes - np.mean(oDataSet.attributes, axis=0))/np.std(oDataSet.attributes, axis=0))
iterations = (len(oDataSet.attributes) * use_train)
time_constant = iterations / np.log(init_radius)

oData = Data(n_class, 11, samples=47)
oData.random_training_test_by_percent([352, 382, 378, 382, 376, 360, 361], use_train)

som = SOM(epochs, n_neurons, oDataSet.attributes.shape[1])
som.cluster_data(oDataSet.attributes[oData.Training_indexes], init_radius, i_learning_rate)

for ep,neurons_matrix in enumerate(som.hist_neurons_matrix):
    labels = np.zeros(neurons_matrix.shape[:2])
    for i in range(neurons_matrix.shape[0]):
        for j in range(neurons_matrix.shape[1]):
            labels[i, j] = 0
            count_label = np.zeros((n_class, 1))
            vizinhos = distance(oDataSet.attributes, neurons_matrix[i, j])
            for k in range(k_nn):
                min = np.argmin(vizinhos)
                count_label[int(oDataSet.labels[min, 0])] += 1
                vizinhos[min] = np.max(vizinhos)
            labels[i, j] = np.argmax(count_label)
    ok = np.zeros((n_class, 1))
    for i in range(n_neurons ** 2):
        x = i // n_neurons
        y = i % n_neurons
        if ok[int(labels[x, y])] == 0:
            plt.scatter(x, y, linewidths=1, color=COLOR[int(labels[x, y])],
                        label="{}".format(oDataSet.labelsNames[int(labels[x, y])]))
            ok[int(labels[x, y])] = 1
        else:
            plt.scatter(x, y, linewidths=1, color=COLOR[int(labels[x, y])])

    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.savefig("Test_{:3d}.png".format(ep), dpi=300, bbox_inches="tight")
    plt.show()


    for i in range(n_neurons ** 2):
        x = i // n_neurons
        y = i % n_neurons
        if ok[int(labels[x, y])] == 0:
            plt.scatter(neurons_matrix[x,y,0], neurons_matrix[x,y,1], linewidths=1, color=COLOR[int(labels[x, y])],
                        label="{}".format(oDataSet.labelsNames[int(labels[x, y])]))
            ok[int(labels[x, y])] = 1
        else:
            plt.scatter(neurons_matrix[x,y,0], neurons_matrix[x,y,1], linewidths=1, color=COLOR[int(labels[x, y])])

    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.savefig("U_{:3d}.png".format(ep), dpi=300, bbox_inches="tight")
    plt.show()

