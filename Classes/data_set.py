from data import *

class DataSet(object):
        """
        Class implemented by Lucas Costa.
        https://www.github.com/lukkascost
        
        Created in 30/10/2017
        Last Modify in 06/11/2017
        
        contact: lucas.costa@lit.ifce.edu.br
        """          
        def __init__(self):
                """
                Contructor 
                """
                self.dataSet = []
                self.length = 0
                self.sum_confusion_matrix = None
                self.atributes = None
                self.labels = None
                self.number_of_samples = 0    
        #----------------------------------------------------------------------
        def append(self, data):
                """                """
                if self.length == 0:
                        self.sum_confusion_matrix = np.zeros(data.confusion_matrix.shape)
                self.sum_confusion_matrix = np.add(self.sum_confusion_matrix,data.confusion_matrix)
                self.dataSet.append(data)            
                self.length += 1
                
        #----------------------------------------------------------------------
        def addSampleOfAtt(self,att):
                """
                """                
                if self.number_of_samples == 0:
                        self.atributes = np.zeros(att[:-1].shape)
                        self.labels = np.zeros(att[-1].shape)                
                self.atributes = np.vstack((self.atributes,att[:-1]))
                self.labels = np.vstack((self.labels,att[-1]))
                if self.number_of_samples == 0:
                        self.atributes = self.atributes[1:]
                        self.labels =   self.labels[1:]                       
                self.number_of_samples += 1  
                return True
        
        #----------------------------------------------------------------------
        def getGeneralAccurace(self):
                """
                """                
                generalAcc = 0
                for i in range(self.sum_confusion_matrix.shape[0]):
                        generalAcc+= self.sum_confusion_matrix[i,i]/(self.length*self.dataSet[0].number_of_testingSamples)
                return generalAcc/self.sum_confusion_matrix.shape[0]
        #----------------------------------------------------------------------
        def exportTestingSamples(self, path , index = 0):
                """
                """                
                atributes = self.atributes[self.dataSet[index].Testing_indexes]
                labels = self.labels[self.dataSet[index].Testing_indexes] 
                np.savetxt(path, np.hstack((atributes,labels)), delimiter=",")
        #----------------------------------------------------------------------
        def exportTrainingSamples(self, path , index = 0):
                """
                """                
                atributes = self.atributes[self.dataSet[index].Training_indexes]
                labels = self.labels[self.dataSet[index].Training_indexes] 
                np.savetxt(path, np.hstack((atributes,labels)), delimiter=",")
        
        #----------------------------------------------------------------------        
        def __str__(self):
                """
                """                
                return str(self.getGeneralAccurace())