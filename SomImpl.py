import time

from cv2 import *
import cv2.ml as ml
from matplotlib import cm

from Classes import *
import numpy as np

import matplotlib.pyplot as plt

from Classes.Cluster.SOM import SOM, distance

COLOR = cm.rainbow(np.linspace(0, 1, 8))

n_neurons = 20
i_learning_rate = 0.3
epochs = 1
n_class = 7
#
# k_nn = 5
#
# use_train = 0.8
#
# att = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=[x for x in range(24)], delimiter=",")
# label = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=-1, dtype=object, delimiter=",")
#
# # initial neighbourhood radius
# init_radius = np.max([n_neurons, n_neurons])
# # radius decay parameter
# exp = Experiment()
# oDataSet = DataSet()
# for j, i in enumerate(att):
#     oDataSet.add_sample_of_attribute(np.array(list(i) + [label[j]]))
# oDataSet.attributes = oDataSet.attributes.astype(float)
# # oDataSet.normalize_data_set()
#
# oDataSet.attributes = np.nan_to_num(
#     (oDataSet.attributes - np.mean(oDataSet.attributes, axis=0)) / np.std(oDataSet.attributes, axis=0))
# iterations = (len(oDataSet.attributes) * use_train)
# time_constant = iterations / np.log(init_radius)
# for n in range(50):
#     oData = Data(n_class, 11, samples=47)
#     oData.random_training_test_by_percent([352, 382, 378, 382, 376, 360, 361], use_train)
#
#     som = SOM(epochs, n_neurons, oDataSet.attributes.shape[1])
#     som.cluster_data(oDataSet.attributes[oData.Training_indexes], init_radius, i_learning_rate)
#
#     som.save("EXP_05.gzip")
#     labels = np.zeros(som.neurons_matrix.shape[:2])
#     for i in range(som.neurons_matrix.shape[0]):
#         for j in range(som.neurons_matrix.shape[1]):
#             labels[i, j] = 0
#             count_label = np.zeros((n_class, 1))
#             vizinhos = distance(oDataSet.attributes, som.neurons_matrix[i, j])
#             for k in range(k_nn):
#                 min = np.argmin(vizinhos)
#                 count_label[int(oDataSet.labels[min, 0])] += 1
#                 vizinhos[min] = np.max(vizinhos)
#             labels[i, j] = np.argmax(count_label)
#
#     ok = np.zeros((n_class, 1))
#     for i in range(n_neurons ** 2):
#         x = i // n_neurons
#         y = i % n_neurons
#         if ok[int(labels[x, y])] == 0:
#             plt.scatter(x, y, linewidths=1, color=COLOR[int(labels[x, y])],
#                         label="{}".format(oDataSet.labelsNames[int(labels[x, y])]))
#             ok[int(labels[x, y])] = 1
#         else:
#             plt.scatter(x, y, linewidths=1, color=COLOR[int(labels[x, y])])
#     oData.model = som
#     plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
#     plt.show()
#     oDataSet.append(oData)
# exp.add_data_set(oDataSet, "{} neuronios e learning de {}".format(n_neurons, i_learning_rate))
# exp.save("Experiment_01")

k_nn = 13
exp = Experiment.load("Experiment_01.gzip")
for oDataSet in exp.experimentResults:
    for oData in oDataSet.dataSet:
        som = oData.model
        labels = np.zeros((oDataSet.attributes[oData.Testing_indexes].shape[0], 1))
        for m, i in enumerate(oDataSet.attributes[oData.Testing_indexes]):
            count_label = np.zeros((n_class, 1))
            vizinhos = distance(som.neurons_matrix, i)
            for k in range(k_nn):
                min = np.argmin(vizinhos)
                count_label[int(som.neurons_labels[min // som.n_neurons, min % som.n_neurons, 0])] += 1
                vizinhos[min // som.n_neurons, min % som.n_neurons] = np.max(vizinhos)
            labels[m, 0] = np.argmax(count_label)
        oData.set_results_from_classifier(labels, oDataSet.labels[oData.Testing_indexes])
        oData.model = som
        oDataSet.sum_confusion_matrix += oData.confusion_matrix
        # print(oData)
    print(oDataSet)
    print(oDataSet.sum_confusion_matrix)

