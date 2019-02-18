# deep-nightmare
What if AI had a mental illness?

### Brief Overview 
This project started out as an experiment to observe the result of passing scary images through a DeepDream model. We wanted to induce nightmares in AI. We experimented with the architecture (Resnet18, Resnet34, Resnet50, Resnet152), the learning rates, the images, data augmentation, and training the model from scratch (no pre-training). These parameters can be set in ```hyperparameters.py```

### Running the Code
This section outlines the basic structure of the code and how to run the model.
##### Preparing the Data
In order to train the model using a particular dataset, please ensure that your data is in the format specified below:
images/
    Class1/
    Class2/
    Class3/
    Class4/
    Test/
The names of the classes can be anything you like, but please ensure that the test images are under a folder titled 'Test'. Since the testing images are the images on which the dreams or nightmares are going to be projected, these images do not (and should not, for the purposes of this experiment) be categorized into the same classes as the training images.
Once the images are in the given format, please run ```utils.py``` as follows:
```python utils.py --data-dir <path-to-folder-containing-images>```
In this case, the path would be ```./images```

The code will split the training images into an 80-20 train-val split and write the paths to txtfiles. These txtfiles will be under a folder named ```data_files```.

##### Training the Model
As mentioned before, the hyperparameters for the network can be set in ```hyperparameters.py```. Once this is done, the model can be trained by simply calling the ```test.py``` file:
```python test.py```
While the model is training, checkpoints will be created for each epoch. At the end of training, the code will print the epochs at which the highest accuracy and lowest loss (both validation) were reached. The user can then enter the epoch number for which the experiments should be generated on the training images.

#### Testing The Model
In the event that there is already a trained model that simply needs to be tested, the path to the checkpoint can be sent in as an argument to ```test.py```:
```python test.py <path-to-checkpoint>```
