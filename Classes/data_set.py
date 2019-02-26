from Classes.data import *


class DataSet(object):
    """
    Class implemented by Lucas Costa.
    https://www.github.com/lukkascost

    Created in 30/10/2017
    Last Modify in 15/08/2018

    contact: lucas.costa@lit.ifce.edu.br
    """

    def __init__(self):
        """
        Constructor
        initializes the variables with zeros or None.
        """
        self.dataSet = []
        self.length = 0
        self.sum_confusion_matrix = None
        self.attributes = None
        self.labels = None
        self.labelsNames = []
        self.number_of_samples = 0
        self.normalize_between = None

    # ----------------------------------------------------------------------
    def append(self, data):
        """
        Parameter data: object of class type Data, with results and confusion matrix complete.
        adds on self object array the values of complete data object.
        """
        if self.length == 0:
            self.sum_confusion_matrix = np.zeros(data.confusion_matrix.shape)
        self.sum_confusion_matrix = np.add(self.sum_confusion_matrix, data.confusion_matrix)
        self.dataSet.append(data)
        self.length += 1

    # ----------------------------------------------------------------------
    def add_sample_of_attribute(self, att):
        """
        Parameter att: array with attributes and label on last position. appends the attribute to object. the
        variable must be composed by a sequence of attributes and the last value is a label of a class which
        attribute belongs.
        """
        if self.number_of_samples == 0:
            self.attributes = np.zeros(att[:-1].shape)
            self.labels = np.zeros(1)
        self.attributes = np.vstack((self.attributes, att[:-1]))
        if att[-1] in self.labelsNames:
            self.labels = np.vstack((self.labels, self.labelsNames.index(att[-1])))
        else:
            self.labelsNames.append(att[-1])
            self.labels = np.vstack((self.labels, self.labelsNames.index(att[-1])))

        if self.number_of_samples == 0:
            self.attributes = self.attributes[1:]
            self.labels = self.labels[1:]

        self.number_of_samples += 1

        return True

    # ----------------------------------------------------------------------
    def get_general_metrics(self):
        """
        gets general metrics for results on confusion matrix, related to all the DATA append on self object.
        returns accuracy, sensitivity and specificity.
        """
        if self.length == 0:
            return False

        metrics = np.array([i.get_metrics() for i in self.dataSet])
        return np.average(metrics, axis=0), np.std(metrics, axis=0)

    def normalize_data_set(self):
        """
        Normalize attributes in self object and saves the values of normalization max and min.
        """
        self.normalize_between = np.zeros((len(self.attributes[0]), 2))
        for i in range(len(self.attributes[0])):
            self.normalize_between[i, 0] = max(self.attributes[:, i])
            self.normalize_between[i, 1] = min(self.attributes[:, i])
            if self.normalize_between[i, 0] - self.normalize_between[i, 1] == 0:
                self.attributes[:, i] = self.attributes[:, i] * 0.0
            else:
                self.attributes[:, i] = self.attributes[:, i] - self.normalize_between[i, 1]
                self.attributes[:, i] = self.attributes[:, i] / (
                        self.normalize_between[i, 0] - self.normalize_between[i, 1])

                # ----------------------------------------------------------------------

    def __str__(self):
        """
        """
        string = "{:#^80}".format(" AVERAGE RESULTS ")
        string += "\n {:^20}\t{:^6}\t{:^6}\t{:^6}\t{:^6}".format("", "acc", "Se", "Es", "F1")
        metrics = None
        for i in range(self.dataSet[0].number_of_classes):
            metrics = self.get_general_metrics()[0]
            string += "\n{:^20}: ".format(self.labelsNames[i])
            string += "\t{:02.04f}".format(metrics[0][i])
            string += "\t{:02.04f}".format(metrics[1][i])
            string += "\t{:02.04f}".format(metrics[2][i])
            string += "\t{:02.04f}".format(metrics[3][i])
        string += "\n{:^20}:".format("All Class")
        string += "\t{:02.04f}".format(metrics[0][-1])
        string += "\t{:02.04f}".format(metrics[1][-1])
        string += "\t{:02.04f}".format(metrics[2][-1])
        string += "\t{:02.04f}\n".format(metrics[3][-1])
        string += "#" * 80

        string += "\n{:#^80}".format(" STD RESULTS ")
        string += "\n {:^20}\t{:^6}\t{:^6}\t{:^6}\t{:^6}".format("", "acc", "Se", "Es", "F1")
        for i in range(self.dataSet[0].number_of_classes):
            metrics = self.get_general_metrics()[1]
            string += "\n{:^20}: ".format(self.labelsNames[i])
            string += "\t{:02.04f}".format(metrics[0][i])
            string += "\t{:02.04f}".format(metrics[1][i])
            string += "\t{:02.04f}".format(metrics[2][i])
            string += "\t{:02.04f}".format(metrics[3][i])
        string += "\n{:^20}:".format("All Class")
        string += "\t{:02.04f}".format(metrics[0][-1])
        string += "\t{:02.04f}".format(metrics[1][-1])
        string += "\t{:02.04f}".format(metrics[2][-1])
        string += "\t{:02.04f}\n".format(metrics[3][-1])
        string += "#" * 80

        return string

    def export_as_latex_table(self, label="tab:dataset", caption="Dataset Table"):
        """
        :param label:
        :param caption:
        :return:
        """
        caption = caption.replace("%", "\\%")
        metrics = None
        string = "\\begin{table}[htb]\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\begin{" \
                                                                                       "center}\n\\begin{" \
                                                                                       "tabular}{@{}lccccc@{" \
                                                                                       "}}\\hline "
        string += "\n{:^20} & {:^6} & {:^6} & {:^6} & {:^6} \\\\\\hline \\hline".format("", "acc", "Se", "Es", "F1")

        for i in range(self.dataSet[0].number_of_classes):
            metrics = self.get_general_metrics()[0]
            string += "\n{:^20} & ".format(self.labelsNames[i])
            string += "\t{:02.04f} &".format(metrics[0][i] * 100)
            string += "\t{:02.04f} &".format(metrics[1][i] * 100)
            string += "\t{:02.04f} &".format(metrics[2][i] * 100)
            string += "\t{:02.04f} \\\\\\hline".format(metrics[3][i] * 100)

        string += "\n{:^20} & ".format('Totals')
        string += "\t{:02.04f} &".format(metrics[0][-1] * 100)
        string += "\t{:02.04f} &".format(metrics[1][-1] * 100)
        string += "\t{:02.04f} &".format(metrics[2][-1] * 100)
        string += "\t{:02.04f} \\\\\\hline".format(metrics[3][-1] * 100)

        string += "\n\\end{tabular}\n\\end{center}\n\\end{table}\n\n"
        return string
