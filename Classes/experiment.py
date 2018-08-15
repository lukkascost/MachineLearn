import gzip
import pickle as pk


class Experiment(object):
    """
        Class implemented by Lucas Costa.
        https://www.github.com/lukkascost

        Created in 30/10/2017
        Last Modify in 15/08/2018

        contact: lucas.costa@lit.ifce.edu.br
        """

    def __init__(self):
        """
                Contructor
                """
        self.experimentResults = []
        self.experimentDescription = []
        self.length = 0

    def add_data_set(self, data_set, description=""):
        """
                Parameter data_set:
                Optional description:
                """
        self.experimentResults.append(data_set)
        self.experimentDescription.append(description)
        self.length += 1

    def __str__(self):
        """
        Overwrite the str conversion to print object:
        """
        result = ""
        for i, j in enumerate(self.experimentResults):
            result += "*" * 40 + self.experimentDescription[i] + "*" * 40 + "\n"
            result += str(j) + "\n"

        return result

    def show_in_table(self):
        """
        :return:
        """
        result = ""
        result += "\n{:^4}\t{:^7}\t{:^7}\t{:^7}\t{:^7}".format("", "acc", "Se", "Es", "F1")
        mean_r = ""
        for i, j in enumerate(self.experimentResults):
            result += "\nR{:02d} \t{:03.04f}".format(i + 1, j.get_general_metrics()[0][0][-1] * 100)
            result += "\t{:03.04f}".format(j.get_general_metrics()[0][1][-1] * 100)
            result += "\t{:03.04f}".format(j.get_general_metrics()[0][2][-1] * 100)
            result += "\t{:03.04f}".format(j.get_general_metrics()[0][3][-1] * 100)
            mean_r += "\nR{:02d} is {}".format(i + 1, self.experimentDescription[i])

        result += "\n\n WHEN: " + mean_r
        return result

    def save(self, filename, protocol=0):
        """
        Saves a compressed object to disk
        Parameter filename:
        Optional Parameter:
        """
        file_res = gzip.GzipFile(filename, 'wb')
        file_res.write(pk.dumps(self, protocol))
        file_res.close()

    @staticmethod
    def load(filename):
        """
        Loads a compressed object from disk
        Parameter filename:
        """
        file_res = gzip.GzipFile(filename, 'rb')
        class_file = ""
        while True:
            data = file_res.read()
            if data == "":
                break
            class_file += data
        result = pk.loads(class_file)
        file_res.close()
        return result
