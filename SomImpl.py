import time

from cv2 import *
import cv2.ml as ml
from matplotlib import cm

from Classes import *
import numpy as np

import matplotlib.pyplot as plt

COLOR = cm.rainbow(np.linspace(0, 1, 5))

mu, sigma = 1, 0.4
mu2, sigma2 = -1, 0.4
qtd = 500

n_neurons = 5
neighboards_test = [4, 3, 2, 1, 0]
learning_rate = 0.1
epochs = 2

s = np.random.normal(mu, sigma, qtd)
s2 = np.random.normal(mu, sigma, qtd)
c1 = np.column_stack((s, s2, np.ones(qtd) * 0))

s3 = np.random.normal(mu2, sigma2, qtd)
s4 = np.random.normal(mu, sigma2, qtd)
c2 = np.column_stack((s3, s4, np.ones(qtd) * 1))

s5 = np.random.normal(mu, sigma2, qtd)
s6 = np.random.normal(mu2, sigma2, qtd)
c3 = np.column_stack((s5, s6, np.ones(qtd) * 2))

s7 = np.random.normal(mu2, sigma2, qtd)
s8 = np.random.normal(mu2, sigma2, qtd)
c4 = np.column_stack((s7, s8, np.ones(qtd) * 3))

data = np.row_stack((c1, c2, c3, c4))

oDataSet = DataSet()
for i in data:
    oDataSet.add_sample_of_attribute(i)

oData = Data(4, int(qtd * .2), samples=qtd)
oData.random_training_test_per_class()
print(oData.Testing_indexes.shape)
print(oData.Training_indexes)


def get_winner(el1, el2):
    d = distance(el1, el2)
    best_element = np.argmin(d)
    return np.array([best_element // el1.shape[0], best_element % el1.shape[1]])


def distance(el1, el2):
    distances = np.zeros(el1.shape[:2])  # (el1 - el2) ** 2
    for i in range(el1.shape[0]):
        for j in range(el1.shape[1]):
            distances[i, j] = ((el2[0] - el1[i, j, 0]) ** 2 + (el2[1] - el1[i, j, 1]) ** 2) ** .5
    return distances


### Criação da matrix de neuronios
neurons_matrix = np.zeros((n_neurons, n_neurons, oDataSet.attributes.shape[1]))
print(neurons_matrix.shape)
for neighboards in neighboards_test:
    for e in range(epochs):
        time.sleep(1)
        np.random.shuffle(oData.Training_indexes)
        for index in oData.Training_indexes:
            winner = get_winner(neurons_matrix, oDataSet.attributes[index])
            for i in range(neurons_matrix.shape[0]):
                for j in range(neurons_matrix.shape[1]):
                    if (winner[0] - neighboards <= i <= winner[0] + neighboards):
                        if (winner[1] - neighboards <= j <= winner[1] + neighboards):
                            neurons_matrix[i, j] = neurons_matrix[i, j] + learning_rate * (
                                    oDataSet.attributes[index] - neurons_matrix[i, j])

        plt.scatter(s, s2, color=COLOR[0])
        plt.scatter(s3, s4, color=COLOR[1])
        plt.scatter(s5, s6, color=COLOR[2])
        plt.scatter(s7, s8, color=COLOR[3])
        plt.scatter(neurons_matrix[:, :, 0].reshape(n_neurons * n_neurons),
                    neurons_matrix[:, :, 1].reshape(n_neurons * n_neurons), linewidths=2, color=COLOR[4])

        for i in range(n_neurons*n_neurons):
            plt.annotate(str(i//n_neurons)+","+ str(i%n_neurons), (neurons_matrix[i//n_neurons, i%n_neurons,0], neurons_matrix[i//n_neurons, i%n_neurons,1]))
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])

        plt.show()


