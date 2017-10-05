from Classes.data import *
from Classes.data_set import DataSet
from Classes.experiment import Experiment
def ler_arquivo(address):
        arquivo = open(address,"r")                                     ##
        bd = []                                                         ##
        obj = 0                                                         ##
        for line in arquivo:                                            ##
                bd.append([])                                           ##
                lines = line.split(",")                                 ##
                for attr in lines:                                      ##
                        if(len(attr)>1): bd[obj].append(float(attr))    ##
                obj = obj+1                                             ##
        return bd                                                       ##


experimento = Experiment()
for percent in [1,2]:
        oDataSet = DataSet()
        bd = ler_arquivo("Classes/Classificators/FEATURES_M{:d}.txt".format(percent))
        for i in bd:
                oDataSet.addSampleOfAtt(i[0:-1], i[-1])
        for it in range(50):
                oData = Data(7, 13)
                oData.confusion_matrix = 13 * np.random.rand(7, 7)
                oDataSet.append(oData, range(259), range(259,350))
        experimento.addDataSet(oDataSet, description="Dataset do M={:d}:".format(percent))
print experimento
                


