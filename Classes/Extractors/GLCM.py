import cv2
import numpy as np

########################################################################
class GLCM:
        """
        Class Contains a GLCM with 24 attributes of an object 
        Class implemented by Lucas Costa.
        https://www.github.com/lukkascost
        
        Created in 30/10/2017
        Last Modify in 26/12/2017
        
        contact: lucas.costa@lit.ifce.edu.br
        """

        #----------------------------------------------------------------------
        def __init__(self, input_array, number_of_bits,number_of_Attributes = 24):
                """
                Generate an square coOccurence Matrix of input_array with shape (2**number_of_bits)
                Parameter input_array:
                Parameter number_of_bits:
                """
                self.coOccurenceMatrix = np.zeros((2**number_of_bits, 2**number_of_bits))
                self.input_array = input_array
                self.coOccurenceNormalized = np.zeros((2**number_of_bits, 2**number_of_bits))
                self.num_att = number_of_Attributes
                self.attributes = np.zeros(number_of_Attributes+1)
                
        def generateCoOccurenceHorizontal(self, step = 1,orientation = True ):
                """
                Calculate the coOccurence Matrix for input value with horizontal neighbor
                optional parameter: step, is a distance in relation a neighbor. default=1
                optional Parameter: orientation, is True if neighbor is on right and False if is on left. dafault=TRUE
                """
                self.coOccurenceMatrix *=0
                for i in range(0,self.input_array.shape[0]):
                        for j in range(0,self.input_array.shape[1]-step,step):
                                self.coOccurenceMatrix[int(self.input_array[i,j]), int(self.input_array[i,j+1])]+= 1
                if not(orientation):
                        self.coOccurenceMatrix = self.coOccurenceMatrix.T
                        
        def normalizeCoOccurence(self, init = 0, endValue = 1):
                """
                Normalize the Occurence matrix with values between 0 and 1.
                optional Parameter: init, is a initial value from normalization, default is 0 
                optional Parameter: endValue, is the final value from normalization, default is 1
                """
                self.coOccurenceNormalized = init + ((endValue*self.coOccurenceMatrix)/(self.input_array.shape[0]*(self.input_array.shape[1] -1))  ) 
                
        def setAtributesValues(self,glcm_att):
                "Parameter glcm_att: Array with attributes"
                self.attributes = glcm_att
                self.num_att = len(glcm_att)
                
                
                
        def calculateAttributes(self):  
                """
                Calculate the 24 attributes of GLCM based on co Occurence matrix,
                the attributes descriptions are:
                01 - Angular Second Moment
                02 - Contrast
                03 - Correlation
                04 - Sum of Squares
                05 - Inverse Difference Moment
                06 - Sum Average
                07 - Sum Variance
                08 - Sum Entropy
                09 - Entropy
                10 - Difference Variance
                11 - Difference entropy
                12 - Information measures of correlation
                13 - Information measures of correlation
                14 - Maximal correlation coefficient
                15 - Homogeneity
                16 - sum Mean
                17 - Maximum Probability
                18 - Cluster Tendency
                19 - Cluster shade
                20 - Cluster prominence
                21 - Dissimilarity
                22 - Difference mean
                23 - Autocorrelation
                24 -Inertia
                """
                gray = self.coOccurenceMatrix.shape[0]
                HXY1 = 0.0
                HXY2 = 0.0
                HX   = 0.0
                HY   = 0.0
                
                px = np.zeros(gray)
                py = np.zeros(gray)
                px_plus_y = np.zeros(gray*2-1)
                px_minus_y = np.zeros(gray)
                for i in range(gray):
                        px[i] = sum(self.coOccurenceNormalized[i,:])
                        py[i] = sum(self.coOccurenceNormalized[:,i])        
                        for j in range(gray):
                                Pij = self.coOccurenceNormalized[i,j]
                                px_plus_y[i+j] += self.coOccurenceNormalized[i,j]
                                px_minus_y[abs(i-j)] += Pij
                                self.attributes[1]  += Pij*Pij
                                self.attributes[2]  += ( (i-j) * (i-j) * (Pij))
                                self.attributes[3]  += (i*j) * Pij
                                self.attributes[5]  += (Pij)/(1+pow(i-j,2))
                                self.attributes[15] += (Pij)/(1+abs(i-j))
                                self.attributes[16] += Pij*(i+j)
                                self.attributes[9]  += Pij* np.log10(Pij+ 1e-30)
                                self.attributes[21] += Pij*abs(i-j)
                                self.attributes[22] += Pij*(i-j)
                                self.attributes[23] += Pij*i*j
                                self.attributes[24] += Pij*pow(i-j,2)
                self.attributes[3]  = (self.attributes[3]  - (np.mean(px)*np.mean(py)))
                self.attributes[3]  /= np.std(px)*np.std(py)
                meanall = (np.mean(px)+np.mean(py))/2
                self.attributes[16]  = self.attributes[16] /2
                self.attributes[22] /= 2
                self.attributes[24] /= pow(pow(gray, 2)-1,2)
                
                Q = np.zeros((gray,gray))

                
                for i in range(gray*2-1):
                        self.attributes[6]  += px_plus_y[i]
                        self.attributes[8]  += px_plus_y[i]*np.log10(px_plus_y[i]+ 1e-30) 
                for i in range(gray):
                        self.attributes[11]  += px_minus_y[i]*np.log10(px_minus_y[i]+ 1e-30) 
                        HX += px[i]*np.log10(px[i]+1e-30)
                        HY += py[i]*np.log10(py[i]+1e-30)        
                        for j in range(gray):
                                HXY1 += self.coOccurenceNormalized[i,j]*np.log10(px[i]*py[j] +1e-30)
                                HXY2 += px[i]*py[j]*np.log10(px[i]*py[j] +1e-30)
                                self.attributes[4] += pow(i-meanall,2)*self.coOccurenceNormalized[i,j]
                                self.attributes[18] += pow(i+j-(2*meanall),2)*self.coOccurenceNormalized[i,j]
                                self.attributes[19] += self.coOccurenceNormalized[i,j] * pow(i+j-np.mean(px)-np.mean(py),3)
                                self.attributes[20] += self.coOccurenceNormalized[i,j] * pow(i+j-np.mean(px)-np.mean(py),4)
                                for k in range(gray):
                                        Q[i,j] = self.coOccurenceNormalized[i,k]*self.coOccurenceNormalized[j,k]
                                        if not(Q[i,j] == 0 ):
                                                Q[i,j] = Q[i,j]/(px[i]*py[k])
                                
                self.attributes[8]  *= -1
                self.attributes[9]  *= -1
                self.attributes[10]  = np.var(px_minus_y)
                self.attributes[11]  *= -1
                HXY1 *= -1
                HXY2 *= -1
                
                self.attributes[12]  = self.attributes[9]  - HXY1
                m1 = np.amax(HX)
                m2 = np.amax(HY)
                self.attributes[17] = np.amax(self.coOccurenceNormalized)
                if m1>m2:
                        self.attributes[12]  /= m1
                else:
                        self.attributes[12]  /= m2
                self.attributes[13]  = pow((1-np.exp(-2.0*(HXY2-self.attributes[9] ))), 1/2)
                
                for i in range(gray*2-1):
                        self.attributes[7]  += pow(i-self.attributes[8] ,2)*px_plus_y[i]
                        
                self.attributes[14] = np.linalg.eig(Q)[0][1]
                self.attributes[14] = pow(self.attributes[14],1/2)
                self.attributes = self.attributes[1:]
                
        def exportToClassfier(self, label):
                """
                Export extractor attributes in a numpy array with a label on last position.
                Parameter label: it's a label to which each attributes belong.
                """
                returnable = np.full((self.num_att+1), "", dtype=object)
                returnable[:-1] = self.attributes
                returnable[-1] = label
                return returnable