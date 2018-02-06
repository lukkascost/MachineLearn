import pickle as pk
import numpy as np
import cv2
import copy
import gzip

class Experiment(object):
        """
        Class implemented by Lucas Costa.
        https://www.github.com/lukkascost
        
        Created in 30/10/2017
        Last Modify in 06/02/2018
        
        contact: lucas.costa@lit.ifce.edu.br
        """        
        def __init__(self):
                """
                Contructor 
                """
                self.experimentResults = []
                self.experimentDescript = []
                self.length = 0
                
        def addDataSet(self,dataSet, description = ""):
                """
                Parameter dataSet: 
                Optional description: 
                """
                self.experimentResults.append(dataSet)
                self.experimentDescript.append(description)
                self.length+=1
        
        def __str__(self):
                """
                Overwrite the str conversion to print object:
                """
                metrics = [x.getGeneralMetrics() for x in self.experimentResults]
                metrics = np.array(metrics)
                avg = metrics[:,0]
                avg = np.average(avg, axis=0)
                std = metrics[:,1] 
                std = np.average(std, axis=0)
                
                string  = "{:#^80}".format(" AVERAGE RESULTS ")
                string += "\n {:^20}\t{:^6}\t{:^6}\t{:^6}\t{:^6}".format("","acc","Se","Es","F1")
                for i in range(self.experimentResults[0].dataSet[0].number_of_classes):
                        string += "\n{:^20}: ".format(self.experimentResults[0].labelsNames[i])
                        string += "\t{:02.04f}".format(avg[0][i])
                        string += "\t{:02.04f}".format(avg[1][i])
                        string += "\t{:02.04f}".format(avg[2][i])
                        string += "\t{:02.04f}".format(avg[3][i])                        
                string += "\n{:^20}:".format("All Class")
                string += "\t{:02.04f}".format(avg[0][-1])
                string += "\t{:02.04f}".format(avg[1][-1])
                string += "\t{:02.04f}".format(avg[2][-1])
                string += "\t{:02.04f}\n".format(avg[3][-1])
                string  += "#"*80   
                
                string += "\n{:#^80}".format(" STD RESULTS ")
                string += "\n {:^20}\t{:^6}\t{:^6}\t{:^6}\t{:^6}".format("","acc","Se","Es","F1")
                for i in range(self.experimentResults[0].dataSet[0].number_of_classes):
                        string += "\n{:^20}: ".format(self.experimentResults[0].labelsNames[i])
                        string += "\t{:02.04f}".format(std[0][i])
                        string += "\t{:02.04f}".format(std[1][i])
                        string += "\t{:02.04f}".format(std[2][i])
                        string += "\t{:02.04f}".format(std[3][i])                        
                string += "\n{:^20}:".format("All Class")
                string += "\t{:02.04f}".format(std[0][-1])
                string += "\t{:02.04f}".format(std[1][-1])
                string += "\t{:02.04f}".format(std[2][-1])
                string += "\t{:02.04f}\n".format(std[3][-1])
                string  += "#"*80 
                return string
        
        def save(self, filename, protocol = 0):
                """
                Saves a compressed object to disk
                Parameter filename: 
                Optional Parameter: 
                """
                fileRes = gzip.GzipFile(filename, 'wb')
                fileRes.write(pk.dumps(self, protocol))
                fileRes.close()        
                
        def load(self,filename):
                """
                Loads a compressed object from disk
                Parameter filename:
                """
                fileRes = gzip.GzipFile(filename, 'rb')
                ClassFile = ""
                while True:
                        data = fileRes.read()
                        if data == "":
                                break
                        ClassFile += data
                result = pk.loads(ClassFile)
                fileRes.close()
                return result