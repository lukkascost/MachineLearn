import cv2
import pickle as pk
import copy

class Experiment(object):
        def __init__(self):
                self.experimentResults = []
                self.experimentDescript = []
                self.length = 0
                
        def addDataSet(self,dataSet, description = ""):
                self.experimentResults.append(dataSet)
                self.experimentDescript.append(description)
                self.length+=1
        
        def __str__(self):
                result = ""
                for i,j in enumerate(self.experimentResults):
                        result += self.experimentDescript[i] + "\n"
                        result += "\t" + str(j) + "\n"
                        
                return result