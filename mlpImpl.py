import numpy as np
from sklearn.metrics import confusion_matrix
import warnings

# Useful functions
from Classes import DataSet, Data, Experiment
from Classes.Classificators.multilayer_perceptron import MLPClassifier_lucas
import matplotlib.pyplot as plt
init = 0.01

X = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", delimiter=',', usecols=[x for x in range(24)])
Y = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", delimiter=',', usecols=-1, dtype=object)

oExp = Experiment()
for learn in range(0, 10, 1):
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
                                  learning_rate_init=init - (learn*0.001),
                                  verbose=False,
                                  power_t=0.5,
                                  max_iter=20000,
                                  shuffle=True,
                                  random_state=21,
                                  tol=0.0001,
                                  momentum=0.9)
        clf.classes_ = np.unique(oDataSet.labels)
        clf.fit(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes])
        # for k in range(20000):
        #     clf = clf.partial_fit(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes])
        #     if clf._no_improvement_count > clf.n_iter_no_change:
        #         print()
        #         break
        # print(k)
        y_pred = clf.predict(oDataSet.attributes[oData.Testing_indexes])
        oData.confusion_matrix = confusion_matrix(oDataSet.labels[oData.Testing_indexes], y_pred)
        oData.params = clf.get_params()
        oData.model = clf
        oDataSet.append(oData)
    oExp.add_data_set(oDataSet, "Experimento variando taxa de aprendixado {} .".format(init - (learn*0.001)))
    oExp.save("Experiment_02_02_MLP_LEARN_BK.gzip")
    oExp.save("Experiment_02_02_MLP_LEARN.gzip")
    print(oExp)
x = []
y = []
y_p = []
y_m = []
METRIC = 1

oExp = Experiment.load("Experiment_02_01_MLP_LEARN.gzip")
for i in oExp.experimentResults:
    x.append(i.dataSet[0].model.learning_rate_init)
    for j in i.dataSet:
        y_pred = j.model.predict(i.attributes[j.Training_indexes])
        j.confusion_matrix = confusion_matrix(i.labels[j.Training_indexes], y_pred)
    y.append(i.get_general_metrics()[0][METRIC, -1])
    y_p.append(i.get_general_metrics()[0][METRIC, -1] + i.get_general_metrics()[1][METRIC, -1])
    y_m.append(i.get_general_metrics()[0][METRIC, -1] - i.get_general_metrics()[1][METRIC, -1])

print(x[-20:-15])
print(y_p[-20:-15])
print(y[-20:-15])

plt.plot(x, y)
plt.scatter(x, y_p)
plt.scatter(x, y_m)
plt.savefig("FIG1/SE_x_LEARN_TRAIN.png".format(), dpi=300, bbox_inches="tight")
plt.show()
