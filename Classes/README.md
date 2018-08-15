# Data Class
text

### ```Data.random_training_test()```
Set in self object the training and testing indexies of atributes list.
Chooses the sets without a specific number of samples of each class, that is the number of samples of each class is random. Some class can`be more samples and others less.


### ```Data.random_training_test_per_class()```
Chooses sets with a specific number of samples from each class. 

For example, if your classifier have 3 classes and 150 samples total for training and 150 for test. 

Each class will have 50 samples for test and 50 samples for training, totalizing 150 samples for training and 150 for test.


### ```Data.set_results_from_classifier()```
text

### ```Data.getMetrics()```
text


### Sample to create a use of Data Class
text


# Data_set Class

### ```Data_set.append(data)```
text


### ```Data_set.add_sample_of_attribute(att)```
text


### ```Data_set.getGeneralAccurace()```
text


### Sample to create a use of DataSet
text


# Experiment Class

### ```Experiment.add_data_set(self,dataSet, description = "")```
text

### ```Experiment.save(self,filename, protocol = 0)```
text

### ```Experiment.load(self,filename)```
text


### Sample to create a use of Experiment
text



## Authors

* **Lucas Costa** - [lukkascost](https://github.com/lukkascost)

See also the list of [contributors](https://github.com/lukkascost/MachineLearn/contributors) who participated in this project.
