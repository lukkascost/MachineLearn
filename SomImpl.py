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

n_neurons = 16
neighboards = 0
learning_rate = 0.2
epochs = 50

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
    return np.argmin(d)


def distance(el1, el2):
    distances = (el1 - el2) ** 2
    return np.sum(distances, axis=1) ** 0.5


### Criação da matrix de neuronios
neurons_matrix = np.zeros((n_neurons, oDataSet.attributes.shape[1]))
print(neurons_matrix.shape)

samples_used = []
for e in range(epochs):
    np.random.shuffle(oData.Training_indexes)
    for index in oData.Training_indexes:
        winner = get_winner(neurons_matrix, oDataSet.attributes[index])
        displacement = learning_rate * (oDataSet.attributes[index] - neurons_matrix[winner])
        neurons_matrix[winner] = neurons_matrix[winner] + displacement


    print("EPOCA ", e)

print(neurons_matrix)

plt.plot(neurons_matrix[:, 0], neurons_matrix[:, 1],  color=COLOR[4])
plt.scatter(s, s2, color=COLOR[0])
plt.scatter(s3, s4, color=COLOR[1])
plt.scatter(s5, s6, color=COLOR[2])
plt.scatter(s7, s8, color=COLOR[3])
plt.scatter(neurons_matrix[:, 0], neurons_matrix[:, 1], linewidths=2,  color=COLOR[4])

plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])

plt.show()
