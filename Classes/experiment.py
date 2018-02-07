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
                result = ""
                for i,j in enumerate(self.experimentResults):
                        result += "*"*40+self.experimentDescript[i] + "*"*40 + "\n"
                        result += str(j) + "\n"

                return result
        
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