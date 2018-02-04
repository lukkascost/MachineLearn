from data import *

class DataSet(object):
        """
        Class implemented by Lucas Costa.
        https://www.github.com/lukkascost
        
        Created in 30/10/2017
        Last Modify in 19/12/2017
        
        contact: lucas.costa@lit.ifce.edu.br
        """          
        def __init__(self):
                """
                Contructor 
                initializes the variables with zeros or None.
                """
                self.dataSet = []
                self.length = 0
                self.sum_confusion_matrix = None
                self.atributes = None
                self.labels = None
                self.labelsNames = []
                self.number_of_samples = 0    
                self.normalize_between = None
        #----------------------------------------------------------------------
        def append(self, data):
                """
                Parameter data: object of class type Data, with results and confusion matrix complete.
                adds on self object array the values of complete data object.                """
                if self.length == 0:
                        self.sum_confusion_matrix = np.zeros(data.confusion_matrix.shape)
                self.sum_confusion_matrix = np.add(self.sum_confusion_matrix,data.confusion_matrix)
                self.dataSet.append(data)            
                self.length += 1
                
        #----------------------------------------------------------------------
        def addSampleOfAtt(self,att):
                """
                Parameter att: array with attributes and label on last position.
                appends the attribute to object.
                the variable must be composed by a sequence of attributes and the last value is a label of a class which attribute belongs.
                """                
                
                if self.number_of_samples == 0:
                        self.atributes = np.zeros(att[:-1].shape)
                        self.labels = np.zeros(att[-1].shape)                
                self.atributes = np.vstack((self.atributes,att[:-1]))
                if (att[-1] in  self.labelsNames):
                        self.labels = np.vstack((self.labels,self.labelsNames.index(att[-1])))
                else:
                        self.labelsNames.append(att[-1])
                        self.labels = np.vstack((self.labels,self.labelsNames.index(att[-1])))
                        
                if self.number_of_samples == 0:
                        self.atributes = self.atributes[1:]
                        self.labels =   self.labels[1:]    
                        
                self.number_of_samples += 1 
                
                return True
        
        #----------------------------------------------------------------------
        def getGeneralMetrics(self):
                """
                gets general metrics for results on confusion matrix, related to all the DATA append on self object.
                returns accurace, sensitivity and specificity.
                """                
                if self.length == 0: return False
                confusion_matrix = self.sum_confusion_matrix/self.length
                
                VP = confusion_matrix.diagonal()
        
                FP = np.sum(confusion_matrix,axis=0)
                FP = FP - VP
        
                FN = np.sum(confusion_matrix,axis=1)
                FN = FN - VP
        
                VN = FP+FN+VP
                VN = (self.dataSet[0].number_of_testingSamples*self.dataSet[0].number_of_classes)-VN
                
                VPP = VP
                VPP = VPP/(VP+VN)
                
                acc = (VP+VN)/(self.dataSet[0].number_of_testingSamples*self.dataSet[0].number_of_classes)
                acc = np.hstack((acc,sum(acc)/self.dataSet[0].number_of_classes))
        
                se = VP / (VP+FN)
                F1 = 2*VPP*se
                F1 = F1/(VPP+se)
                
                se = np.hstack((se,sum(se)/self.dataSet[0].number_of_classes))
        
                es = VN/(VN+FP)
                es = np.hstack((es,sum(es)/self.dataSet[0].number_of_classes))
                
                return np.array([acc,se,es,F1])
        
        def normalizeDataSet(self):
                """
                Normalize atributes in self object and saves the values of normalization max and min.
                """
                self.normalize_between = np.zeros((len(self.atributes[0]),2))
                for i in range(len(self.atributes[0])):
                        self.normalize_between[i,0] = max(self.atributes[:,i])
                        self.normalize_between[i,1] = min(self.atributes[:,i])              
                        if self.normalize_between[i,0]-self.normalize_between[i,1] == 0:
                                self.atributes[:,i] = self.atributes[:,i] * 0.0
                        else:
                                self.atributes[:,i] = self.atributes[:,i] - self.normalize_between[i,1]
                                self.atributes[:,i] = self.atributes[:,i] / (self.normalize_between[i,0] - self.normalize_between[i,1])                                                    
                                       
        #----------------------------------------------------------------------        
        def __str__(self):
                """
                """       
                string  = "{:#^80}".format(" AVERAGE RESULTS ")
                string += "\n {:^20}\t{:^6}\t{:^6}\t{:^6}\t{:^6}".format("","acc","Se","Es","F1")
                for i in range(self.dataSet[0].number_of_classes):
                        metrics = self.getGeneralMetrics()
                        string += "\n{:^20}: ".format(self.labelsNames[i])
                        string += "\t{:02.04f}".format(metrics[0][i])
                        string += "\t{:02.04f}".format(metrics[1][i])
                        string += "\t{:02.04f}".format(metrics[2][i])
                        string += "\t{:02.04f}".format(metrics[3][i])                        
                string += "\n{:^20}:".format("All Class")
                string += "\t{:02.04f}".format(metrics[0][-1])
                string += "\t{:02.04f}".format(metrics[1][-1])
                string += "\t{:02.04f}".format(metrics[2][-1])
                string += "\t{:02.04f}\n".format(metrics[3][-1])
                
                
                
                string  += "#"*80
                
                return string 