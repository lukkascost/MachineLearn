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
                self.Testing_indexes = None
                self.Training_indexes = None
        #----------------------------------------------------------------------
        def randomTrainingTest(self,startClass = 1):
                """
                Optional Parameter startClass: number of first class, default=1
                """
                allIndexes = np.arange(self.number_of_classes*(self.number_of_testingSamples+self.number_of_trainingSamples))
                np.random.shuffle(allIndexes)
                self.Testing_indexes = allIndexes[:self.number_of_classes * self.number_of_trainingSamples]
                self.Training_indexes = allIndexes[self.number_of_classes * self.number_of_trainingSamples:]
                return True        
        def __str__(self):
                return ""