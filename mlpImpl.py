import numpy as np
from sklearn.metrics import confusion_matrix
import warnings

# Useful functions
from Classes import DataSet, Data
from Classes.Classificators.multilayer_perceptron import MLPClassifier_lucas

X = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", delimiter=',', usecols=[x for x in range(24)])
Y = np.loadtxt("../Fluxo_experiments/GLCM/EXP_01/FEATURES_M1_CM8b.txt", delimiter=',', usecols=-1, dtype=object)

oDataSet = DataSet()

for j, i in enumerate(X):
    oDataSet.add_sample_of_attribute(np.array(list(i) + [Y[j]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
# oDataSet.normalize_data_set()
oDataSet.attributes = np.nan_to_num(
    (oDataSet.attributes - np.mean(oDataSet.attributes, axis=0)) / np.std(oDataSet.attributes, axis=0))

oData = Data(7, 10)

oData.random_training_test_by_percent([352, 382, 378, 382, 376, 360, 361], 0.8)
clf = MLPClassifier_lucas(hidden_layer_sizes=(50,),
                    activation="tanh",
                    solver='sgd',
                    alpha=0.0001,
                    batch_size=200,
                    learning_rate="constant",
                    learning_rate_init=0.001,
                    verbose=True,
                    power_t=0.5,
                    max_iter=20000,
                    shuffle=True,
                    random_state=21,
                    tol=0.00001,
                    momentum=0.9)
clf.classes_ = np.unique(oDataSet.labels)
for i in range(30000):
    clf = clf.partial_fit(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes])
    if clf._no_improvement_count > clf.n_iter_no_change:
        break
    print(clf.loss, clf.warm_start, clf.early_stopping, clf._no_improvement_count)
y_pred = clf.predict(oDataSet.attributes[oData.Testing_indexes])
oData.confusion_matrix = confusion_matrix(oDataSet.labels[oData.Testing_indexes], y_pred)
print(oData)
