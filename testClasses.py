from cv2 import *
import cv2.ml as ml
from Classes import *
import numpy as np

#def ler_arquivo(address):
        #arquivo = open(address,"r")                                     ##
        #bd = []                                                         ##
        #obj = 0                                                         ##
        #for line in arquivo:                                            ##
                #bd.append([])                                           ##
                #lines = line.split(",")                                 ##
                #for attr in lines:                                      ##
                        #if(len(attr)>1): bd[obj].append(float(attr))    ##
                #obj = obj+1                                             ##
        #return bd                                                       ##


#experimento = Experiment()
#for percent in [1,2]:
        #oDataSet = DataSet()
        #bd = ler_arquivo("Classes/Classificators/FEATURES_M{:d}.txt".format(percent))
        #for i in bd:
                #oDataSet.add_sample_of_attribute(np.array(i[0:-1]), np.array(i[-1]))
        #for it in range(50):
                #oData = Data(7, 13)
                #oData.confusion_matrix = 13 * np.random.rand(7, 7)
                #a = np.arange(350)
                #np.random.shuffle(a)
                #t = a[:37*7]
                #b= []
                #for i in a:
                        #if not i in t:
                                #b.append(i)                    
                #oDataSet.append(oData, t, b)
        #experimento.add_data_set(oDataSet, description="Dataset do M={:d}:".format(percent))
#print experimento

#entrada = [[2,1,3,0],[0,1,1,3],[1,3,1,2],[0,1,0,2]]

#np_entrada  = np.array(entrada)

#print np_entrada
#print np.vstack((np_entrada,np_entrada[0]))


#gl = GLCM(np_entrada, 2)

#gl.generateCoOccurenceHorizontal()
#gl.normalizeCoOccurence()
#gl.calculateAttributes()




#for nbits in [8]:
        #arrayGLCM = np.zeros((0,25))            
        #for i in [4]:
                #fname = "../isolador-multiclasse/base/tempo/filtrado_lucas/{}b/Classe_{}.txt.gz".format(nbits,i)
                #array = np.loadtxt(fname, delimiter=",")
                #array[139,1477] = 255
                
                #print "Creating array of objects..."
                #glArray = np.array([GLCM(np.matrix(x), nbits) for x in np.loadtxt(fname, delimiter=",")],dtype="object")
                
                #print "generate co occurence matrix..."
                #[gl.generateCoOccurenceHorizontal() for gl in glArray]
                
                #print "Normalizing co occurence ..."
                #[gl.normalizeCoOccurence() for gl in glArray]
                
                #print "Calculating attributes..."
                #[gl.calculateAttributes() for gl in glArray]
                
                #print "Exporting attributes"
                #for gl in glArray:
                        #arrayGLCM = np.vstack((arrayGLCM,gl.exportToClassfier(float(i+1)))) 
                
                #print "Finish!"
                
                #print arrayGLCM
                #print "\n"
        #np.savetxt("GLCM_FILES/{}b.txt.gz".format(nbits),arrayGLCM,delimiter = ",", fmt="%.10e")
        #print "File Generated successful"


#for nbits in [5,6,7,8]:
        #fname = "../SVM_ISOLADOR/GLCM_FILES/{}b.txt.gz".format(nbits)
        #array = np.loadtxt(fname, delimiter=",")
        #obDataSet = DataSet()
        #for j in range(1,6):
                #ar = array[array[:,-1]==j]
                #np.random.shuffle(ar)
                #ar = ar[:200]
                #for i in ar:
                        #obDataSet.add_sample_of_attribute(i)
        ##obDataSet.normalize_data_set()
        
        
        #for itIndex in range(niterations):
                #obData = Data(5, 50, samples=200)
                #obData.radndomTrainingTestPerClass()
                #svm = cv2.SVM()
                #obData.params = dict(kernel_type = cv2.SVM_RBF,svm_type = cv2.SVM_C_SVC,gamma=2.0,nu = 0.0,p = 0.0, coef0 = 0)
                #svm.train(np.float32(obDataSet.attributes[obData.Training_indexes]),np.float32(obDataSet.labels[obData.Training_indexes]),None,None,obData.params)
                #svm.save("asd.txt")
                #results =  svm.predict_all(np.float32(obDataSet.attributes[obData.Testing_indexes]),np.float32(obDataSet.labels[obData.Testing_indexes]))
                #obData.set_results_from_classifier(results, obDataSet.labels[obData.Testing_indexes])
                #obDataSet.append(obData)
        #exp.add_data_set(obDataSet, description="Test for {}bits database: ".format(nbits))
        #exp.save("file.txt")
        #print exp
#print exp
#labelNames = [""]

oDataSet = DataSet()
oDataSet.add_sample_of_attribute(np.array([1.1, 1, 1, 1, 1, 1, 'First_Class']))
oDataSet.add_sample_of_attribute(np.array([2.1, 2, 2, 2, 2, 2, 'Second_Class']))
oDataSet.add_sample_of_attribute(np.array([1.1, 1, 1, 1, 1, 1, 'First_Class']))
oDataSet.add_sample_of_attribute(np.array([2.1, 2, 2, 2, 2, 2, 'Second_Class']))
oDataSet.add_sample_of_attribute(np.array([1.1, 1, 1, 1, 1, 1, 'First_Class']))
oDataSet.add_sample_of_attribute(np.array([2.1, 2, 2, 2, 2, 2, 'Second_Class']))
oDataSet.add_sample_of_attribute(np.array([1.1, 1, 1, 1, 1, 1, 'First_Class']))
oDataSet.add_sample_of_attribute(np.array([2.1, 2, 2, 2, 2, 2, 'Second_Class']))
oDataSet.add_sample_of_attribute(np.array([1.1, 1, 1, 1, 1, 1, 'First_Class']))
oDataSet.add_sample_of_attribute(np.array([2.1, 2, 2, 2, 2, 2, 'Second_Class']))
oDataSet.add_sample_of_attribute(np.array([1.1, 1, 1, 1, 1, 1, 'First_Class']))
oDataSet.add_sample_of_attribute(np.array([2.1, 2, 2, 2, 2, 2, 'Second_Class']))

oData = Data(2, 2, samples=6)
oData.random_training_test_by_percent(np.array([6, 6]), 0.50)

svm = ml.SVM_create()
svm = svm.create()
svm.trainAuto(np.float32(oDataSet.attributes[oData.Training_indexes]), ml.ROW_SAMPLE, np.int32(oDataSet.labels[oData.Training_indexes]), kFold=2)
oData.insert_model(svm)
print (oData.model)
results = []
for i in (oDataSet.attributes[oData.Testing_indexes]):
    res,cls = svm.predict(np.float32([i]))
    print (res, cls)
    results.append(cls[0])

oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
oDataSet.append(oData)
print(oData)

