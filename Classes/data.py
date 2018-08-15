import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

class Data(object):
    """
        Class implemented by Lucas Costa.
        https://www.github.com/lukkascost

        Created in 30/10/2017
        Last Modify in 15/08/2018

        contact: lucas.costa@lit.ifce.edu.br
        """

    def __init__(self, n_class, n_testing_samples, samples=50):
        """
        Parameter n_class: number of classes to be classify.
        Parameter n_testing_samples: number of samples for test in each class.
        Optional Parameter samples: number of samples for each class, default 50.
        """
        self.confusion_matrix = np.zeros((n_class, n_class))
        self.number_of_classes = n_class
        self.number_of_trainingSamples = samples - n_testing_samples
        self.number_of_testingSamples = n_testing_samples
        self.Testing_indexes = None
        self.Training_indexes = None
        self.model = None
        self.params = None

    def random_training_test_by_percent(self, quantity_per_class, percent_train):
        """
        Set in self object the training and testing indexies of attributes list.
        Chooses the sets with a specific percent of number of samples of each class.
        """
        self.Testing_indexes = []
        self.Training_indexes = []
        array = np.arange(np.sum(quantity_per_class))

        for i, j in enumerate(quantity_per_class[:-1], start=1):
            quantity_per_class[i] += j
        array = np.split(array, quantity_per_class[:-1])
        [np.random.shuffle(x) for x in array]
        for i in array:
            self.Training_indexes += list(i[:int(len(i) * percent_train)])
            self.Testing_indexes += list(i[int(len(i) * percent_train):])
        self.Testing_indexes = np.array(self.Testing_indexes)
        self.Training_indexes = np.array(self.Training_indexes)

    def random_training_test(self):
        """
        Set in self object the training and testing indexies of attributes list.
        Chooses the sets without a specific number of samples of each class.
        """
        all_indexes = np.arange(
            self.number_of_classes * (self.number_of_testingSamples + self.number_of_trainingSamples))
        np.random.shuffle(all_indexes)
        self.Testing_indexes = all_indexes[self.number_of_classes * self.number_of_trainingSamples:]
        self.Training_indexes = all_indexes[:self.number_of_classes * self.number_of_trainingSamples]
        return True

    def random_training_test_per_class(self):
        """
        Set in self object the training and testing indexies of attributes list.
        Chooses sets with a specific number of samples from each class.
        """
        self.Testing_indexes = []
        self.Training_indexes = []
        array = np.arange(self.number_of_classes * (self.number_of_testingSamples + self.number_of_trainingSamples))
        array = np.split(array, self.number_of_classes)
        [np.random.shuffle(x) for x in array]
        for i in array:
            self.Testing_indexes = self.Testing_indexes + list(i[:self.number_of_testingSamples])
            self.Training_indexes += list(i[self.number_of_testingSamples:])
        self.Testing_indexes = np.array(self.Testing_indexes)
        self.Training_indexes = np.array(self.Training_indexes)

    def set_results_from_classifier(self, results, labels):
        """
        Parameter results: array with results of classfier.
        Parameter labels: label for each entry in classfier.
        """
        for i, j in enumerate(results):
            self.confusion_matrix[int(labels[i, 0]), int(j[0])] += 1

    def get_metrics(self):
        """
        returns the metrics of accuracy, sensitivity and specificity of classifier.
        """
        v_p = self.confusion_matrix.diagonal()

        f_p = np.sum(self.confusion_matrix, axis=0)
        f_p = f_p - v_p

        f_n = np.sum(self.confusion_matrix, axis=1)
        f_n = f_n - v_p

        v_n = f_p + f_n + v_p
        v_n = (np.sum(self.confusion_matrix)) - v_n

        v_p_p = v_p / (v_p + f_p)
        v_p_p[np.isnan(v_p_p)] = 0.0

        acc = (v_p + v_n) / (np.sum(self.confusion_matrix))
        acc = np.hstack((acc, sum(acc) / self.number_of_classes))

        se = v_p / (v_p + f_n)
        f1 = 2 * v_p_p * se
        f1 = f1 / (v_p_p + se)
        f1[np.isnan(f1)] = 0.0

        se = np.hstack((se, sum(se) / self.number_of_classes))
        f1 = np.hstack((f1, sum(f1) / self.number_of_classes))

        es = v_n / (v_n + f_p)
        es = np.hstack((es, sum(es) / self.number_of_classes))

        return np.nan_to_num(np.array([acc, se, es, f1]))

    def insert_model(self, model, path="tmp.txt"):
        """

        :param model:
        :param path:
        :return:
        """
        model.save(path)
        self.model = open(path, 'r').read()
        if path == "tmp.txt":
            os.remove("tmp.txt")

    def save_model(self, path):
        """

        :param path:
        :return:
        """
        if self.model is None:
            raise Exception("Model not inserted!\n insert the model before to save him.")
        open(path, "w").write(self.model)

    def __str__(self):
        string = "\n" + "\t" * 3 + "acc\tSe\tEs\tF1"
        metrics = None
        i = 0
        for i in range(self.number_of_classes):
            metrics = self.get_metrics()
            string += "\nClass {} metrics: ".format(i + 1)
            string += "\t{:02.04f}".format(metrics[0][i])
            string += "\t{:02.04f}".format(metrics[1][i])
            string += "\t{:02.04f}".format(metrics[2][i])
            string += "\t{:02.04f}".format(metrics[3][i])
        string += "\nAll Class metrics: ".format(i + 1)
        string += "\t{:02.04f}".format(metrics[0][-1])
        string += "\t{:02.04f}".format(metrics[1][-1])
        string += "\t{:02.04f}".format(metrics[2][-1])
        string += "\t{:02.04f}".format(metrics[3][-1])

        return string
