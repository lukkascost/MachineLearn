import numpy as np
import itertools

def calc_y_desc(data, coefs):
    a = 1 / (-coefs[0] + coefs[1])
    b = 1 + (coefs[0] / (-coefs[0] + coefs[1]))
    return (-a) * data + b


def calc_y_asc(data, coefs):
    a = 1 / (-coefs[0] + coefs[1])
    b = (-coefs[0] / (-coefs[0] + coefs[1]))
    return (a) * data + b


data = np.loadtxt("iris.data", usecols=[0, 1, 2, 3], delimiter=",")
labels = np.loadtxt("iris.data", usecols=-1, delimiter=",", dtype=object)
data_fuzzy = np.zeros(data.shape, dtype=object)
data_pert = np.zeros((150, 4, 5))
NOMES = ['0', '1', '2', '3', '4']

coefs_all = [[4.3, 5.02, 5.74, 6.46, 7.18, 7.9],
             [2.0, 2.48, 2.96, 3.44, 3.92, 4.4],
             [1.0, 2.18, 3.3600000000000003, 4.540000000000001, 5.720000000000001, 6.9],
             [0.1, 0.58, 1.06, 1.54, 2.02, 2.5]]

for att in range(4):
    coefs = coefs_all[att]
    filtro = np.bitwise_and(data[:, att] >= coefs[0], data[:, att] < coefs[1])
    data_pert[filtro, att, 0] = calc_y_desc(data[filtro, att], [coefs[0], coefs[1]])
    for i in range(1, 5):
        filtro = np.bitwise_and(data[:, att] >= coefs[i - 1], data[:, att] < coefs[i])
        data_pert[filtro, att, i] = calc_y_asc(data[filtro, att], [coefs[i - 1], coefs[i]])
        filtro = np.bitwise_and(data[:, att] >= coefs[i], data[:, att] < coefs[i + 1])
        data_pert[filtro, att, i] = calc_y_desc(data[filtro, att], [coefs[i], coefs[i + 1]])
    i = 5
    filtro = np.bitwise_and(data[:, att] >= coefs[i - 1], data[:, att] < coefs[i])
    data_pert[filtro, att, i - 1] = calc_y_asc(data[filtro, att], [coefs[i - 1], coefs[i]])

for k in range(data.shape[0]):
    for l in range(data.shape[1]):
        data_fuzzy[k, l] = NOMES[np.argmax(data_pert[k, l])]

# att = 0
#
# filtro = labels == 'Iris-setosa'
# unique, counts = np.unique(data_fuzzy[filtro, att], return_counts=True)
# print(dict(zip(unique, counts)))
#
# filtro = labels == 'Iris-versicolor'
# unique, counts = np.unique(data_fuzzy[filtro, att], return_counts=True)
# print(dict(zip(unique, counts)))
#
# filtro = labels == 'Iris-virginica'
# unique, counts = np.unique(data_fuzzy[filtro, att], return_counts=True)
# print(dict(zip(unique, counts)))
#
combinations = np.array([p for p in itertools.product(NOMES, repeat=4)])
defuzzy = dict()

for k in combinations:
    filtro = np.logical_and(np.logical_and(data_fuzzy[:, 0] == k[0], data_fuzzy[:, 1] == k[1]), np.logical_and(data_fuzzy[:, 2] == k[2], data_fuzzy[:, 3] == k[3]))
    unique, counts = np.unique(labels[filtro], return_counts=True)
    if(len(unique) == 0):
        defuzzy[k[0]+k[1]+k[2]+k[3]] = 'NAO CLASSIFICAVEL'
    else:
        defuzzy[k[0] + k[1] + k[2] + k[3]] = unique[np.argmax(counts)]

def defu(defuzzy,data):
    return defuzzy[data[0]+data[1]+data[2]+data[3]]

print(data_fuzzy[0])
results = [defu(defuzzy, x) for x in data_fuzzy]
results = np.array(results)
print(results)
print(data_fuzzy[results == 'NAO CLASSIFICAVEL'])