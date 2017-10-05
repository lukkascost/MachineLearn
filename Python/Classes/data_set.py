from data import *

class DataSet(object):
        def __init__(self):
                self.dataSet = []
                self.length = 0
                self.indexSet = []
                self.sum_confusion_matrix = None
                self.atributes = []
                self.labels = []
                self.number_of_samples = 0                 
        def append(self, data, traineIndexes, testIndexes):
                if self.length == 0:
                        self.sum_confusion_matrix = np.zeros(data.confusion_matrix.shape)
                self.sum_confusion_matrix = np.add(self.sum_confusion_matrix,data.confusion_matrix)
                self.dataSet.append(data)
                self.indexSet.append([traineIndexes,testIndexes])
                self.length += 1
        def addSampleOfAtt(self,att):
                self.atributes.append(att[:-1])
                self.labels.append(att[-1])
                self.number_of_samples += 1
        def addSampleOfAtt(self,att,label):
                self.atributes.append(att)
                self.labels.append(label)
                self.number_of_samples += 1  
        def getGeneralAccurace(self):
                generalAcc = 0
                for i in range(self.sum_confusion_matrix.shape[0]):
                        generalAcc+= self.sum_confusion_matrix[i,i]/(self.length*self.dataSet[0].number_of_testingSamples)
                return generalAcc/self.sum_confusion_matrix.shape[0]
                        
        def __str__(self):
                return str(self.getGeneralAccurace())