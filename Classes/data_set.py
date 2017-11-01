from data import *

class DataSet(object):
        def __init__(self):
                self.dataSet = []
                self.length = 0
                self.indexSet = []
                self.sum_confusion_matrix = None
                self.atributes = None
                self.labels = None
                self.number_of_samples = 0                 
        def append(self, data, traineIndexes, testIndexes):
                if self.length == 0:
                        self.sum_confusion_matrix = np.zeros(data.confusion_matrix.shape)
                self.sum_confusion_matrix = np.add(self.sum_confusion_matrix,data.confusion_matrix)
                self.dataSet.append(data)
                if self.length == 0:
                        self.indexSet = np.zeros(np.array([traineIndexes,testIndexes]).shape)
                self.indexSet = np.vstack((self.indexSet, np.array([traineIndexes,testIndexes])))
                if self.length == 0:
                        self.indexSet = self.indexSet[1:]                
                self.length += 1
                
        def addSampleOfAtt(self,att):
                if self.number_of_samples == 0:
                        self.atributes = np.zeros(att[:-1].shape)
                        self.labels = np.zeros(att[-1].shape)                
                self.atributes = np.vstack((self.atributes,att[:-1]))
                self.labels = np.vstack((self.labels,att[-1]))
                if self.number_of_samples == 0:
                        self.atributes = self.atributes[1:]
                        self.labels =   self.labels[1:]                       
                self.number_of_samples += 1  
                
        def addSampleOfAtt(self,att,label):
                if self.number_of_samples == 0:
                        self.atributes = np.zeros(att.shape)
                        self.labels = np.zeros(label.shape)                
                self.atributes = np.vstack((self.atributes,att))
                self.labels = np.vstack((self.labels,label))
                if self.number_of_samples == 0:
                        self.atributes = self.atributes[1:]
                        self.labels =   self.labels[1:]                       
                self.number_of_samples += 1  
                
        def getGeneralAccurace(self):
                generalAcc = 0
                for i in range(self.sum_confusion_matrix.shape[0]):
                        generalAcc+= self.sum_confusion_matrix[i,i]/(self.length*self.dataSet[0].number_of_testingSamples)
                return generalAcc/self.sum_confusion_matrix.shape[0]
        def exportTestingSamples(self, path , index = 0):
                atributes = self.atributes[self.indexSet[index,1]]
                labels = self.labels[self.indexSet[index,1]] 
                np.savetxt(path, np.hstack((atributes,labels)), delimiter=",")
                           
        def exportTrainingSamples(self, path , index = 0):
                atributes = self.atributes[self.indexSet[index,0]]
                labels = self.labels[self.indexSet[index,0]] 
                np.savetxt(path, np.hstack((atributes,labels)), delimiter=",")
        
                
        def __str__(self):
                return str(self.getGeneralAccurace())