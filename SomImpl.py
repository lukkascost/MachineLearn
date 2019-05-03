import time

from cv2 import *
import cv2.ml as ml
from matplotlib import cm

from Classes import *
import numpy as np

import matplotlib.pyplot as plt

COLOR = cm.rainbow(np.linspace(0, 1, 8))

# mu, sigma = 1, 0.4
# mu2, sigma2 = -1, 0.4
# qtd = 500

n_neurons = 40
i_learning_rate = 0.01
epochs = 10
n_class = 7

k_nn = 5

use_train = 0.8

# s = np.random.normal(mu, sigma, qtd)
# s2 = np.random.normal(mu, sigma, qtd)
# c1 = np.column_stack((s, s2, np.ones(qtd) * 0))
#
# s3 = np.random.normal(mu2, sigma2, qtd)
# s4 = np.random.normal(mu, sigma2, qtd)
# c2 = np.column_stack((s3, s4, np.ones(qtd) * 1))
#
# s5 = np.random.normal(mu, sigma2, qtd)
# s6 = np.random.normal(mu2, sigma2, qtd)
# c3 = np.column_stack((s5, s6, np.ones(qtd) * 2))
#
# s7 = np.random.normal(mu2, sigma2, qtd)
# s8 = np.random.normal(mu2, sigma2, qtd)
# c4 = np.column_stack((s7, s8, np.ones(qtd) * 3))

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
print(oData.Testing_indexes.shape)
print(oData.Training_indexes)


def get_winner(el1, el2):
    d = distance(el1, el2)
    best_element = np.argmin(d)
    return np.array([best_element // el1.shape[0], best_element % el1.shape[1]])


def distance(el1, el2):
    return np.sqrt(np.sum((el2 - el1) ** 2, axis=-1))


def calculate_influence(distance, radius):
    return np.exp(-distance / (2 * (radius ** 2)))


def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)


def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)


neurons_matrix = np.zeros((n_neurons, n_neurons, oDataSet.attributes.shape[1]))
print(neurons_matrix.shape)
for ep in range(epochs):
    np.random.shuffle(oData.Training_indexes)
    print(ep)
    for itera, index in enumerate(oData.Training_indexes):
        winner = get_winner(neurons_matrix, oDataSet.attributes[index])
        neighboards = decay_radius(init_radius, itera, time_constant)
        learning_rate = decay_learning_rate(i_learning_rate, itera, iterations)

        for i in range(neurons_matrix.shape[0]):
            for j in range(neurons_matrix.shape[1]):
                w_dist = np.sum((np.array([i, j]) - winner) ** 2)
                w_dist = np.sqrt(w_dist)
                if w_dist <= neighboards:
                    influence = calculate_influence(w_dist, neighboards)
                    neurons_matrix[i, j] = neurons_matrix[i, j] + learning_rate * influence * (
                            oDataSet.attributes[index] - neurons_matrix[i, j])
                    # print(neighboards, influence, learning_rate, winner, [i,j])

# plt.scatter(s, s2, color=COLOR[0])
# plt.scatter(s3, s4, color=COLOR[1])
# plt.scatter(s5, s6, color=COLOR[2])
# plt.scatter(s7, s8, color=COLOR[3])
# plt.scatter(neurons_matrix[:, :, 0].reshape(n_neurons * n_neurons),
#             neurons_matrix[:, :, 1].reshape(n_neurons * n_neurons), linewidths=2, color=COLOR[4])
#
# for i in range(n_neurons * n_neurons):
#     plt.annotate(str(i // n_neurons) + "," + str(i % n_neurons), (
#         neurons_matrix[i // n_neurons, i % n_neurons, 0], neurons_matrix[i // n_neurons, i % n_neurons, 1]))
# plt.xlim([-2.5, 2.5])
# plt.ylim([-2.5, 2.5])
#
# plt.show()

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


