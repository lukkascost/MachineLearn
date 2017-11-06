import cv2
import pickle as pk
import numpy as np
import copy
import gzip

class Data(object):
        """
        Class implemented by Lucas Costa.
        https://www.github.com/lukkascost
        
        Created in 30/10/2017
        Last Modify in 06/11/2017
        
        contact: lucas.costa@lit.ifce.edu.br
        """          
        def __init__(self,nClass, nTestingSamples, samples = 50):
                """
                Constructor 
                """
                self.confusion_matrix = np.zeros((nClass,nClass))
                self.number_of_classes = nClass
                self.number_of_trainingSamples = samples - nTestingSamples
                self.number_of_testingSamples = nTestingSamples
                self.Testing_indexes = None
                self.Training_indexes = None
                self.params = None
        #----------------------------------------------------------------------
        def randomTrainingTest(self,startClass = 1):
                """
                Optional Parameter startClass: number of first class, default=1
                """
                allIndexes = np.arange(self.number_of_classes*(self.number_of_testingSamples+self.number_of_trainingSamples))
                np.random.shuffle(allIndexes)
                self.Testing_indexes = allIndexes[self.number_of_classes * self.number_of_trainingSamples:]
                self.Training_indexes = allIndexes[:self.number_of_classes * self.number_of_trainingSamples]
                return True   
        #----------------------------------------------------------------------
        def setResultsFromClassfier(self,results, labels):
                """
                Parameter results: array with results of classfier.
                Parameter labels: label for each entry in classfier.
                """
                for i, j in enumerate(results):
                        self.confusion_matrix[int(labels[i,0])-1,int(j[0])-1] += 1
                
        def __str__(self):
                return ""