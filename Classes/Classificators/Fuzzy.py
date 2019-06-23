import itertools

import numpy as np


class Fuzzy():
    def __init__(self, NOMES, coefs_all):
        self.NOMES = NOMES
        self.coefs_all = coefs_all
        self.defuzzy = dict()

    def calc_y_desc(self,data, coefs):
        a = 1 / (-coefs[0] + coefs[1])
        b = 1 + (coefs[0] / (-coefs[0] + coefs[1]))
        return (-a) * data + b

    def calc_y_asc(self, data, coefs):
        a = 1 / (-coefs[0] + coefs[1])
        b = (-coefs[0] / (-coefs[0] + coefs[1]))
        return (a) * data + b

    def train(self, data, labels):

        data_fuzzy = self.fuzzy_data(data)
        combinations = np.array([p for p in itertools.product(self.NOMES, repeat=data.shape[1])])

        for k in combinations:
            filtro = np.logical_and(np.logical_and(data_fuzzy[:, 0] == k[0], data_fuzzy[:, 1] == k[1]),
                                    np.logical_and(data_fuzzy[:, 2] == k[2], data_fuzzy[:, 3] == k[3]))
            unique, counts = np.unique(labels[filtro], return_counts=True)
            if len(unique) == 0:
                self.defuzzy[k[0] + k[1] + k[2] + k[3]] = 3
            else:
                self.defuzzy[k[0] + k[1] + k[2] + k[3]] = unique[np.argmax(counts)]

    def predict(self, data):
        return self.defuzzy[data[0] + data[1] + data[2] + data[3]]

    def fuzzy_data(self, data):
        data_fuzzy = np.zeros(data.shape, dtype=object)
        data_pert = np.zeros((data.shape[0], data.shape[1], len(self.NOMES)))

        for att in range(data.shape[1]):
            coefs = self.coefs_all[att]
            filtro = np.bitwise_and(data[:, att] >= coefs[0], data[:, att] < coefs[1])
            data_pert[filtro, att, 0] = self.calc_y_desc(data[filtro, att], [coefs[0], coefs[1]])
            for i in range(1, len(self.NOMES)):
                filtro = np.bitwise_and(data[:, att] >= coefs[i - 1], data[:, att] < coefs[i])
                data_pert[filtro, att, i] = self.calc_y_asc(data[filtro, att], [coefs[i - 1], coefs[i]])
                filtro = np.bitwise_and(data[:, att] >= coefs[i], data[:, att] < coefs[i + 1])
                data_pert[filtro, att, i] = self.calc_y_desc(data[filtro, att], [coefs[i], coefs[i + 1]])
            i = len(self.NOMES)
            filtro = np.bitwise_and(data[:, att] >= coefs[i - 1], data[:, att] < coefs[i])
            data_pert[filtro, att, i - 1] = self.calc_y_asc(data[filtro, att], [coefs[i - 1], coefs[i]])

        for k in range(data.shape[0]):
            for l in range(data.shape[1]):
                data_fuzzy[k, l] = self.NOMES[np.argmax(data_pert[k, l])]
        return data_fuzzy