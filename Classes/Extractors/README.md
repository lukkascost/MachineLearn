# MachineLearn

This project is a base, for the others machine learn and image process projects.

## GLCM Class for extract atributes

### to generate a sample object use:
```
from Classes.Extractors.GLCM import GLCM
gl = GLCM(matrix, number_of_bits)
```
```gl``` is an object from GLCM with follow values:
* coOccurenceMatrix : calculated from method ```generateCoOccurenceHorizontal```
* coOccurenceNormalized : calculated from method ```normalizeCoOccurence```
* input_array : input_array is a parameter from contructor with a bi-dimensional matrix
* attributes  :  calculated from method ```calculate_attributes```

### ```generateCoOccurenceHorizontal```
to set values on the ```gl.coOccurenceMatrix``` use the method ```generateCoOccurenceHorizontal```
this method Calculate the coOccurence Matrix for input value with horizontal neighbor.

optional parameter: step, is a distance in relation a neighbor. default=1.

optional Parameter: orientation, is True if neighbor is on right and False if is on left. dafault=TRUE.

sample of use:
```
from Classes.Extractors.GLCM import GLCM
gl = GLCM(np.array([[2,1,3,0],[0,1,1,3],[1,3,1,2],[0,1,0,2]]), 2)
gl = GLCM(np_entrada, 2)
gl.generateCoOccurenceHorizontal()
```



### ```normalizeCoOccurence```

Normalize the Occurence matrix with values between 0 and 1.

optional Parameter: init, is a initial value from normalization, default is 0 
                
optional Parameter: endValue, is the final value from normalization, default is 1
                
sample of use:
```
from Classes.Extractors.GLCM import GLCM
gl = GLCM(np.array([[2,1,3,0],[0,1,1,3],[1,3,1,2],[0,1,0,2]]), 2)
gl = GLCM(np_entrada, 2)
gl.generateCoOccurenceHorizontal()
gl.normalizeCoOccurence()
```


### ```calculateAttributes```
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

equations in paper: **Reducing costs of embedded image classfier** - [Paper](https://github.com/lukkascost)


sample of use:

```
from Classes.Extractors.GLCM import GLCM
gl = GLCM(np.array([[2,1,3,0],[0,1,1,3],[1,3,1,2],[0,1,0,2]]), 2)
gl = GLCM(np_entrada, 2)
gl.generateCoOccurenceHorizontal()
gl.normalizeCoOccurence()
gl.calculateAttributes()
```

### ```exportToClassfier```

Export extractor attributes in a numpy array with a label on last position.

Parameter label: it's a label to which each attributes belong.

sample of use:

```
from Classes.Extractors.GLCM import GLCM
gl = GLCM(np.array([[2,1,3,0],[0,1,1,3],[1,3,1,2],[0,1,0,2]]), 2)
gl = GLCM(np_entrada, 2)
gl.generateCoOccurenceHorizontal()
gl.normalizeCoOccurence()
gl.calculateAttributes()
print(gl.exportToClassfier(1.0))
```

as Result:

```
[  1.38888889e-01   2.83333333e+00   8.51049772e+01   1.81250000e+00
   3.83333333e-01   1.00000000e+00   5.70242068e+00   5.89762289e-01
   9.09729266e-01   2.77777778e-02   4.96706242e-01   3.44268165e-01
   1.00000000e+00   0.00000000e+00   4.51388889e-01   1.33333333e+00
   2.50000000e-01   6.08333333e+00   1.87916667e+01   6.06458333e+01
   1.50000000e+00  -1.66666667e-01   1.41666667e+00   1.25925926e-02
   1.00000000e+00]
```
## LBP



   
## Authors

* **Lucas Costa** - [lukkascost](https://github.com/lukkascost)

See also the list of [contributors](https://github.com/lukkascost/MachineLearn/contributors) who participated in this project.
