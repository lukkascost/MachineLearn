import cv2
import pickle as pk
import copy
import gzip

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
        
        def save(self, filename, protocol = 0):
                """Saves a compressed object to disk
                """
                fileRes = gzip.GzipFile(filename, 'wb')
                fileRes.write(pk.dumps(self, protocol))
                fileRes.close()        
                
        def load(self,filename):
                """Loads a compressed object from disk
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