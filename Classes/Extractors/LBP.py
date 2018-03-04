import numpy as np


class Lbp8Bits:
    """"""
    def __init__(self, input_array):
        self.input_array = input_array
        self.att58 = [1, 2, 3, 4, 5, 7, 8, 9, 13, 15, 16, 17, 25, 29, 31, 32, 33, 49, 57, 61, 63, 64, 65,
                      97, 113, 121, 125, 127, 128, 129, 130, 132, 136, 144, 160, 192, 193, 194, 196, 200,
                      208, 224, 225, 226, 228, 232, 240, 241, 242, 244, 248, 249, 250, 252, 253, 254, 255, 256]
        self.number_of_attributes = 257
        self.histogram = np.zeros(self.number_of_attributes)

    def calculate_attributes(self):
        """"""
        for i in range(1, self.input_array.shape[0] - 1):
            for j in range(1, self.input_array.shape[1] - 1):
                central = self.input_array[i, j]
                p11 = int(self.input_array[i - 1, j - 1] < central)
                p12 = int(self.input_array[i - 1, j] < central)
                p13 = int(self.input_array[i - 1, j + 1] < central)
                p21 = int(self.input_array[i, j - 1] < central)
                p23 = int(self.input_array[i, j + 1] < central)
                p31 = int(self.input_array[i + 1, j - 1] < central)
                p32 = int(self.input_array[i + 1, j] < central)
                p33 = int(self.input_array[i + 1, j + 1] < central)
                output = p11 + p21 * 2 + p31 * 4 + p32 * 8 + p33 * 16 + p23 * 32 + p13 * 64 + p12 * 128
                self.histogram[output] += 1
                if not (output in self.att58):
                    self.histogram[-1] += 1

    def set_attributes_values(self, lbp_att):
        """"""
        self.histogram = lbp_att
        self.number_of_attributes = len(lbp_att)

    def export_to_classifier(self, label):
        """"""
        returnable = np.full((self.number_of_attributes + 1), "", dtype=object)
        returnable[:-1] = self.histogram
        returnable[-1] = label
        return returnable


class Lbp4Bits:
    """"""
    def __init__(self, input_array):
        self.input_array = input_array
        self.number_of_attributes = 16
        self.histogram = np.zeros(self.number_of_attributes)

    def calculate_attributes(self):
        """"""
        for i in range(1, self.input_array.shape[0] - 1):
            for j in range(1, self.input_array.shape[1] - 1):
                central = self.input_array[i, j]
                p12 = int(self.input_array[i - 1, j] < central)
                p21 = int(self.input_array[i, j - 1] < central)
                p23 = int(self.input_array[i, j + 1] < central)
                p32 = int(self.input_array[i + 1, j] < central)
                output = p21 + p32 * 2 + p23 * 4 + p12 * 8
                self.histogram[output] += 1

    def set_attributes_values(self, lbp_att):
        """"""
        self.histogram = lbp_att
        self.number_of_attributes = len(lbp_att)

    def export_to_classifier(self, label):
        """"""
        returnable = np.full((self.number_of_attributes + 1), "", dtype=object)
        returnable[:-1] = self.histogram
        returnable[-1] = label
        return returnable
