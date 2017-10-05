import cv2
import pickle as pk
import numpy as np
import copy

class Data(object):
        def __init__(self,nClass, nTestingSamples, samples = 50):
                self.confusion_matrix = np.zeros((nClass,nClass))
                self.number_of_classes = nClass
                self.number_of_trainingSamples = samples - nTestingSamples
                self.number_of_testingSamples = nTestingSamples
        def __str__(self):
                return ""