# deep-nightmare
What if AI had a mental illness?

### Brief Overview
This project started out as an experiment to observe the result of passing scary images through a DeepDream model. We wanted to induce nightmares in AI. We experimented with the architecture (Resnet18, Resnet34, Resnet50, Resnet152), the learning rates, the images, data augmentation, and training the model from scratch (no pre-training). These parameters can be set in ```hyperparameters.py```

### Dataset
The Improved Dataset of Nightmare Images (ImPN) and can be found [here](https://drive.google.com/open?id=1EnJhKBbtGkBVtwfRpsD3h22Nfi9Lj3JQ).

The Animals and Nightmare Faces (ANF) can be found [here](https://drive.google.com/open?id=1VubGXc3kbOx6S-Ir-VkOx1cO3Pjq3E3u).

The final testing images are also inside ImPN.

### Running the Code
This section outlines the basic structure of the code and how to run the model.
#### Preparing the Data:
There are two kinds of data that this project handles: binary classes (ImPN) and multi-class labels (ANF).
#### ImPN
ImPN was created with the idea of grouping all images into nightmares and not-nightmares categories. The images were further categorized based on their quality and resemblance to the class they fall under. Thus, the 'Class A' images are the top-grade images, followed by Class B and Class C.
Once the images are downloaded, please ensure they are in the follow directory structure:

    images/

      Class1/

      Class2/

      Class3/

      Class4/

      Final Test Images/

In the above example, Class1, Class2, etc. refer to the original folder names in the dataset. Please do not rename these folders. You may, however, rename the parent directory, i.e., 'images/'.
The final testing images are the same for both ImPN and ANF. Since these are the images on which the dreams or nightmares are going to be projected, these images do not (and should not, for the purposes of this experiment) be categorized into the same classes as the training images.
Once the images are in the given format, please run ```utils.py``` as follows:
```
python utils.py --data-dir <path-to-folder-containing-images> --ANF 0
```
In this case, the path would be ```./images```. ```--ANF``` is set to 0 since we are using the ImPN dataset.

The code will create a new directory called 'data' and move the classes to the directory, while also renaming the images. It will split the training images into an 80-20 train-val split *for each set of grade images*, i.e., it will create a trainA.txt, trainB.txt etc. and write the paths to text files. These text files will be under a folder named ```data_files```.

You can set the list of grades in ```hyperparameters.py``` you want to use for an experiment. During data loading, depending on the grades specified, new train and validation text files will be created using a combination of the grades and their corresponding image folders.

##### ANF
Once the ANF dataset is downloaded, you can run ```utils.py``` like before with a slight modification:
```
python utils.py --data-dir <path-to-folder-containing-images> --ANF 1
```
Once again, please ensure that the path you give is the direct parent directory containing the multiple classes.

When using the ANF dataset, set the 'ANIMALS' parameter in ```hyperparameters.py``` to True.

#### Training the Model
As mentioned before, the hyperparameters for the network can be set in ```hyperparameters.py```. Once this is done, the model can be trained by simply calling the ```test.py``` file:
```
python test.py
```

While the model is training, checkpoints will be created for each epoch. At the end of training, the code will print the epochs at which the highest accuracy and lowest loss (both validation) were reached. The user can then enter the epoch number for which the experiments should be generated on the training images.

#### Testing The Model
In the event that there is already a trained model that simply needs to be tested, the path to the checkpoint can be sent in as an argument to ```test.py```:
```
python test.py <path-to-checkpoint>
```
