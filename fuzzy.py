import numpy as np
import itertools

from Classes import Experiment, DataSet, Data
from Classes.Classificators.Fuzzy import Fuzzy

data = np.loadtxt("iris.data", usecols=[0, 1, 2, 3], delimiter=",")
labels = np.loadtxt("iris.data", usecols=-1, delimiter=",", dtype=object)
NOMES = ['0', '1', '2', '3', '4']
coefs_all = [[4.3, 5.02, 5.74, 6.46, 7.18, 7.9],
             [2.0, 2.48, 2.96, 3.44, 3.92, 4.4],
             [1.0, 2.18, 3.3600000000000003, 4.540000000000001, 5.720000000000001, 6.9],
             [0.1, 0.58, 1.06, 1.54, 2.02, 2.5]]

oExp = Experiment()
for learn in range(0, 1, 1):
    oDataSet = DataSet()
    for j, i in enumerate(data):
        oDataSet.add_sample_of_attribute(np.array(list(i) + [labels[j]]))
    oDataSet.attributes = oDataSet.attributes.astype(float)
    oDataSet.labelsNames.append("SEM INFORMACAO")
    # oDataSet.attributes = np.nan_to_num(
    #     (oDataSet.attributes - np.mean(oDataSet.attributes, axis=0)) / np.std(oDataSet.attributes, axis=0))

    for iteration in range(50):
        oData = Data(4, 10)

        oData.random_training_test_by_percent([50, 50, 50], 0.8)
        fuzzy = Fuzzy(NOMES, coefs_all)
        fuzzy.train(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes])
        for x in oData.Testing_indexes:
            oData.confusion_matrix[int(fuzzy.predict(fuzzy.fuzzy_data(np.array([oDataSet.attributes[x]]))[0]))][int(oDataSet.labels[x])]+= 1
        oData.model = fuzzy
        oDataSet.append(oData)
    oExp.add_data_set(oDataSet, "Experimento variando Com 10 atributos .".format())
    oExp.save("Experiment_03_01_MLP_BK.gzip")
    oExp.save("Experiment_03_01_MLP.gzip")
    print(oExp)
    print(oDataSet.sum_confusion_matrix/50)
