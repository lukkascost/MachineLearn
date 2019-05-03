import numpy as np


class Data(object):
    def __init__(self,  path, delimiter=","):
        self.pure_data = np.loadtxt(path, delimiter=delimiter)
        self.attributes = self.pure_data[:, :-1]
        self.pure_labels = self.pure_data[:, -1]

