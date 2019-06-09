import numpy as np
from sklearn.metrics import confusion_matrix
import warnings

# Useful functions
from Classes import DataSet, Data, Experiment
from Classes.Classificators.multilayer_perceptron import MLPClassifier_lucas
import matplotlib.pyplot as plt
init = 0.01

basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
basemask = basemask - 1

X = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", delimiter=',', usecols=basemask)
Y = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", delimiter=',', usecols=-1, dtype=object)

oExp = Experiment()
for learn in range(0, 1, 1):
    oDataSet = DataSet()

    for j, i in enumerate(X):
        oDataSet.add_sample_of_attribute(np.array(list(i) + [Y[j]]))
    oDataSet.attributes = oDataSet.attributes.astype(float)
    # oDataSet.normalize_data_set()
    oDataSet.attributes = np.nan_to_num(
        (oDataSet.attributes - np.mean(oDataSet.attributes, axis=0)) / np.std(oDataSet.attributes, axis=0))

    for iteration in range(30):
        print(learn ,init - (learn*0.001), iteration)
        oData = Data(7, 10)

        oData.random_training_test_by_percent([352, 382, 378, 382, 376, 360, 361], 0.8)
        clf = MLPClassifier_lucas(hidden_layer_sizes=(18,),
                                  activation="tanh",
                                  solver='sgd',
                                  alpha=0.0001,
                                  batch_size=200,
                                  learning_rate="constant",
                                  learning_rate_init=0.05,
                                  verbose=True,
                                  power_t=0.5,
                                  max_iter=20000,
                                  shuffle=True,
                                  random_state=21,
                                  tol=0.0001,
                                  momentum=0.9)
        clf.classes_ = np.unique(oDataSet.labels)
        # clf.fit(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes])
        for k in range(20000):
            clf = clf.partial_fit(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes])
            if clf._no_improvement_count > clf.n_iter_no_change:
                print()
                break
            y_pred, _y = clf.predict(oDataSet.attributes[oData.Testing_indexes])
            # plt.clf()
            # plt.scatter([x for x in range(len(oDataSet.attributes[oData.Testing_indexes]))],_y[:,6])
            # plt.scatter([x for x in range(len(oDataSet.attributes[oData.Testing_indexes]))],_y[:,0])
            # plt.savefig("FIG2/{:04d}.png".format(k), dpi=201, bbox_inches="tight")
        oData.confusion_matrix = confusion_matrix(oDataSet.labels[oData.Testing_indexes], y_pred)
        oData.params = clf.get_params()
        oData.model = clf
        oDataSet.append(oData)
    oExp.add_data_set(oDataSet, "Experimento variando Com 10 atributos .".format(init - (learn*0.001)))
    oExp.save("Experiment_03_01_MLP_BK.gzip")
    oExp.save("Experiment_03_01_MLP.gzip")
    print(oExp)
x = []
y = []
y_p = []
y_m = []
METRIC = 0

oExp = Experiment.load("Experiment_03_01_MLP.gzip")

oDataSet = oExp.experimentResults[0]

print(oDataSet)
print(oDataSet.sum_confusion_matrix/30)

