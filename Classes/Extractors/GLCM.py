import cv2
import numpy as np

########################################################################
class GLCM:
        """Class Contains a GLCM with 24 atributes of an object 
        """

        #----------------------------------------------------------------------
        def __init__(self, input_array, number_of_bits):
                """Generate an square coOccurence Matrix of input_array with shape (2**number_of_bits)"""
                self.coOccurenceMatrix = np.zeros((2**number_of_bits, 2**number_of_bits))
                self.input_array = input_array
                self.coOccurenceNormalized = np.zeros((2**number_of_bits, 2**number_of_bits))
        def generateCoOccurenceHorizontal(self, step = 1,orientation = True ):
                """Calculate the coOccurence Matrix for input value with horizontal neighbor
                optional parameter: step, is a distance in relation a neighbor. default=1
                optional Parameter: orientation, is True if neighbor is on right and False if is on left. dafault=TRUE
                """
                self.coOccurenceMatrix *=0
                for i in range(0,self.input_array.shape[0]):
                        for j in range(0,self.input_array.shape[1]-step,step):
                                self.coOccurenceMatrix[self.input_array[i,j], self.input_array[i,j+1]]+= 1
                if not(orientation):
                        self.coOccurenceMatrix = self.coOccurenceMatrix.T
                        
        def normalizeCoOccurence(self, init = 0, endValue = 1):
                """Normalize the Occurence matrix with values between 0 and 1.
                optional Parameter: init, is a initial value from normalization, default is 0 
                optional Parameter: endValue, is the final value from normalization, default is 1
                """
                for i in range(self.coOccurenceMatrix.shape[0]):                                                                                              
                        for j in range(self.coOccurenceMatrix.shape[0]):                                                                                      
                                self.coOccurenceNormalized[i,j] = init + ((endValue*self.coOccurenceMatrix[i,j])/(self.input_array.shape[0]*(self.input_array.shape[1] -1))  )                  