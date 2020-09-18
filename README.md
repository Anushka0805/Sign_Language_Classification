# Sign Language Classification
Welcome to our project. The main motivation behind this project is the benefit it provides by bridging the gaps in communication for people with hearing or speech related disabilities. The project aims to successfully translate input images of sign language gestures to the respective alphabets or numbers.

![American Sign Language](/outputs/asl.png)

We used a dataset (https://www.kaggle.com/ahmedkhanak1995/sign-language-gesture-images-dataset) from Kaggle. It consists of various American Sign Language gestures. It adds up to 55500 images belonging to **37 classes** of alphabets, digits and " _ ".
The model has also been successfully implemented on Streamlit, an open source Python library which is used for building custom web apps and deploying machine learning models. The Streamlit app has been demonstrated below:

![American Sign Language App Demo](/outputs/ASL_app_demo.mp4)

- We used the images in the folder called 'Gesture Image Pre-Processed Data' which consists of masked and preprocessed images of dimensions (50,50,3)
- The data was split into necessary parts; training, validation and testing, so that the model can be evaluated correctly.
- A 3 layer convolutional neural network was used to build the model. This architecture along with a fully connected dense layer was fed to the output layer which used *softmax* activation function consisting of 37 units.
- The model was then compiled using the 'adam' optimizer. The model, with a batch size of 128, was trained on 44,000 images for 10 epochs. It resulted in a training accuracy of 0.9148 and validation accuracy of 0.9995.
- The model was also evaluated on the testing data. It churned out an accuracy of 0.9996.
- Images of some of the gestures were obtained from Google. They were resized, masked and fed as inputs to the model for predictions. The results (model's predictions) along with the various images that were used, are given below.


![output 1](/outputs/1.png)
![output 2](/outputs/2.png)
![output 2](/outputs/3.png)
![output 2](/outputs/4.png)
![output 2](/outputs/5.png)

## To run the code:
Download the dataset from the Kaggle link and unzip it.

In Sign_Language_Classification.py, provide the path to 'Gesture Image Pre-Processed Data' in 'inpath' and set 'outpath' to the path to the new folder for storing all the images in one place. (We have called the folder 'dataset').  

Make sure SL_model.h5 is in the same folder as the rest of the files and the new folder.

Next, provide the path to the image you want to test and set 'path' to it.

Run Sign_Language_Classification.py
