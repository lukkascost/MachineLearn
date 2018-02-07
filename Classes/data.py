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
        Last Modify in 06/02/2018
        
        contact: lucas.costa@lit.ifce.edu.br
        """          
        def __init__(self,nClass, nTestingSamples, samples = 50):
                """
                Parameter nClass: number of classes to be classify.
                Parameter nTestingSamples: number of samples for test in each class.
                Optional Parameter samples: number of samples for each class, default 50.
                """
                self.confusion_matrix = np.zeros((nClass,nClass))
                self.number_of_classes = nClass
                self.number_of_trainingSamples = samples - nTestingSamples
                self.number_of_testingSamples = nTestingSamples
                self.Testing_indexes = None
                self.Training_indexes = None
                self.params = None
        #----------------------------------------------------------------------
        def randomTrainingTestByPercent(self, quantity_per_class,percentTrain):
                """
                Set in self object the training and testing indexies of atributes list.
                Chooses the sets with a specific percent of number of samples of each class.
                """
                self.Testing_indexes = []
                self.Training_indexes = []  
                array = np.arange(np.sum(quantity_per_class))
                array = np.split(array, quantity_per_class[:-1])
                [np.random.shuffle(x) for x in array]
                for i in array:
                        self.Training_indexes += list(i[:int(len(i)*percentTrain)])
                        self.Testing_indexes += list(i[int(len(i)*percentTrain):])
                self.Testing_indexes  = np.array(self.Testing_indexes)
                self.Training_indexes  = np.array(self.Training_indexes)                
        
        def randomTrainingTest(self):
                """
                Set in self object the training and testing indexies of atributes list.
                Chooses the sets without a specific number of samples of each class.
                """
                allIndexes = np.arange(self.number_of_classes*(self.number_of_testingSamples+self.number_of_trainingSamples))
                np.random.shuffle(allIndexes)
                self.Testing_indexes = allIndexes[self.number_of_classes * self.number_of_trainingSamples:]
                self.Training_indexes = allIndexes[:self.number_of_classes * self.number_of_trainingSamples]
                return True   
        #----------------------------------------------------------------------
        def randomTrainingTestPerClass(self):
                """
                Set in self object the training and testing indexies of atributes list.
                Chooses sets with a specific number of samples from each class.
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
                        self.confusion_matrix[int(labels[i,0]),int(j[0])] += 1

        #----------------------------------------------------------------------
        def getMetrics(self):
                """
                returns the metrics of accuracy, sensitivity and specificity of classifier.
                """
                VP = self.confusion_matrix.diagonal()
                
                FP = np.sum(self.confusion_matrix,axis=0)
                FP = FP - VP
                
                FN = np.sum(self.confusion_matrix,axis=1)
                FN = FN - VP
                
                VN = FP+FN+VP
                VN = (np.sum(self.confusion_matrix))-VN
                
        
                VPP = VP/(VP+FP)
                
                acc = (VP+VN)/(np.sum(self.confusion_matrix))
                acc = np.hstack((acc,sum(acc)/self.number_of_classes))
                        
                se = VP / (VP+FN)
                F1 = 2*VPP*se
                F1 = F1/(VPP+se)
                
                se = np.hstack((se,sum(se)/self.number_of_classes))
                F1 = np.hstack((F1,sum(F1)/self.number_of_classes))
                
                
                es = VN/(VN+FP)
                es = np.hstack((es,sum(es)/self.number_of_classes))
                
                return np.array([acc,se,es,F1])
        def __str__(self):
                string = "\n"+ "\t"*3 + "acc\tSe\tEs\tF1"
                for i in range(self.number_of_classes):
                        metrics = self.getMetrics()
                        string += "\nClass {} metrics: ".format(i+1)
                        string += "\t{:02.04f}".format(metrics[0][i])
                        string += "\t{:02.04f}".format(metrics[1][i])
                        string += "\t{:02.04f}".format(metrics[2][i])
                        string += "\t{:02.04f}".format(metrics[3][i])                        
                string += "\nAll Class metrics: ".format(i+1)
                string += "\t{:02.04f}".format(metrics[0][-1])
                string += "\t{:02.04f}".format(metrics[1][-1])
                string += "\t{:02.04f}".format(metrics[2][-1])
                string += "\t{:02.04f}".format(metrics[3][-1])                
                
                return string