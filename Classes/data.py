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
        def randomTrainingTest(self):
                """
                Optional Parameter startClass: number of first class, default=1
                """
                allIndexes = np.arange(self.number_of_classes*(self.number_of_testingSamples+self.number_of_trainingSamples))
                np.random.shuffle(allIndexes)
                self.Testing_indexes = allIndexes[self.number_of_classes * self.number_of_trainingSamples:]
                self.Training_indexes = allIndexes[:self.number_of_classes * self.number_of_trainingSamples]
                return True   
        #----------------------------------------------------------------------
        def radndomTrainingTestPerClass(self, startClass=1):
                """
                Optional Parameter startClass: number of first class, default=1
                """
                self.Testing_indexes = []
                self.Training_indexes = []                
                array = np.arange(self.number_of_classes*(self.number_of_testingSamples+ self.number_of_trainingSamples))
                array = np.split(array, self.number_of_classes)
                [np.random.shuffle(x) for x in array]
                for i in array:
                        self.Testing_indexes = self.Testing_indexes + list(i[:self.number_of_testingSamples])
                        self.Training_indexes += list(i[self.number_of_testingSamples:])
                self.Testing_indexes  = np.array(self.Testing_indexes)
                self.Training_indexes  = np.array(self.Training_indexes)
                
                        
                        
        #----------------------------------------------------------------------
        def setResultsFromClassfier(self,results, labels):
                """
                Parameter results: array with results of classfier.
                Parameter labels: label for each entry in classfier.
                """
                for i, j in enumerate(results):
                        self.confusion_matrix[int(labels[i,0])-1,int(j[0])-1] += 1
        def getAccuracePerClass(self):
                """ """
                acuraces = np.zeros(self.number_of_classes)
                for i in range(self.number_of_classes):
                        acuraces[i] = self.confusion_matrix[i,i]/sum(self.confusion_matrix[i,:])
                return acuraces
        def getAccuraceAllClass(self):
                return sum(self.getAccuracePerClass())/self.number_of_classes
        def __str__(self):
                return ""