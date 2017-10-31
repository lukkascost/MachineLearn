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
01 - 

02 - 

03 - 

04 -

05 - 

06 - 

07 - 

08 -

09 -

10 - 

11 -

12 - 

13 - 

14 - 

15 - 

16 - 

17 - 

18 - 

19 - 

20 - 

21 - 

22 - 

23 - 

24 -
            
sample of use:

```
from Classes.Extractors.GLCM import GLCM
gl = GLCM(np.array([[2,1,3,0],[0,1,1,3],[1,3,1,2],[0,1,0,2]]), 2)
gl = GLCM(np_entrada, 2)
gl.generateCoOccurenceHorizontal()
gl.normalizeCoOccurence()
gl.calculate_atributes()
```




   
## Authors

* **Lucas Costa** - [lukkascost](https://github.com/lukkascost)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

