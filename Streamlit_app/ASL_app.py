#import necessary modules
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from keras.preprocessing.image import img_to_array,array_to_img
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import string

st.set_option('deprecation.showfileUploaderEncoding', False)

#main page details
st.title('Sign Language Gesture Recognition Model')
st.subheader("Here, you can input the images of various American sign language gestures to view the model's respective predictions.")

#create a menu
menu=['Welcome','Description','App']
c=st.sidebar.selectbox('Menu',menu)
if c=='Welcome':
    st.subheader('Welcome to our project')
    st.write('The main motivation behind this project is the benefit it provides by bridging the gaps in communication for people with hearing or speech related disabilities.')
    st.write('The project aims to successfully translate input images of sign language gestures to the respective alphabets or numbers.')
    st.write('The model has been trained on alphabets and digits using a dataset from Kaggle and will hence make predictions accordingly.')
    st.write("Read more about the model in the 'Description' section.")
elif c=='Description':
    st.title('Model description')
    st.write("A 3 layer convolutional neural network was used to build the model. This architecture along with a fully connected dense layer was fed to the output layer which used softmax activation function consisting of 37 units.")
    st.write("The model was then compiled using the 'adam' optimizer. The model, with a batch size of 128, was trained on 44,000 images for 10 epochs. It resulted in a training accuracy of 0.9148 and validation accuracy of 0.9995.")
    st.write("The model was also evaluated on the testing data. It churned out an accuracy of 0.9996.")
    st.write("Since the model was trained on augmented images of few people performing all the gestures in similar backgorunds, the predictions on real world data may not be entirely accurate.")
    st.write("The model can be tested by uploading an image corresponding to any gesture in the 'App' section. The prediction will be displayed instantly.")
elif c=='App':
    st.title('Welcome to the working model.')
    up_file=st.file_uploader("Please upload your input image",type=['jpg','png'])

    #define a function to read input images and display predictions
    def image_stuff(img1):
        img=np.asarray(img1)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0, 0, 120])
        upper_blue = np.array([180, 38, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img, img, mask=mask)
        b, g, r = cv2.split(result)
        filter = g.copy()
        ret,mask = cv2.threshold(filter,10,255, 1)
        resized = cv2.resize(mask, (50,50))/255.0
        test_img=img_to_array(resized)
        eg = np.ndarray(shape=(1, 50, 50, 1),dtype=np.float32)
        eg[0]=test_img

        #load the model
        model=tf.keras.models.load_model('/content/MIC_project.h5')
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.load_weights('/content/MIC_project.h5')
        new_pred=model.predict(eg[:1])
        st.write('Running predictions...')
        ynew=np.argmax(new_pred, axis=-1)

        #assign the classes
        if(ynew>=10):
            if (ynew==10):
                ynew='A'
            elif (ynew==11):
                ynew='B'
            elif (ynew==12):
                ynew='C'
            elif (ynew==13):
                ynew='D'
            elif (ynew==14):
                ynew='E'
            elif (ynew==15):
                ynew='F'
            elif (ynew==16):
                ynew='G'
            elif (ynew==17):
                ynew='H'
            elif (ynew==18):
                ynew='I'
            elif (ynew==19):
                ynew='J'
            elif (ynew==20):
                ynew='K'
            elif (ynew==21):
                ynew='L'
            elif (ynew==22):
                ynew='M'
            elif (ynew==23):
                ynew='N'
            elif (ynew==24):
                ynew='O'
            elif (ynew==25):
                ynew='P'
            elif (ynew==26):
                ynew='Q'
            elif (ynew==27):
                ynew='R'
            elif (ynew==28):
                ynew='S'
            elif (ynew==29):
                ynew='T'
            elif (ynew==30):
                ynew='U'
            elif (ynew==31):
                ynew='V'
            elif (ynew==32):
                ynew='W'
            elif (ynew==33):
                ynew='X'
            elif (ynew==34):
                ynew='Y'
            elif (ynew==35):
                ynew='Z'
            elif (ynew==36):
                ynew='Unknown'
            st.write("Model's prediction: "+ynew)
        elif(ynew<10):
            if (ynew==0):
                ynew='0'
            elif (ynew==1):
                ynew='1'
            elif (ynew==2):
                ynew='2'
            elif (ynew==3):
                ynew='3'
            elif (ynew==4):
                ynew='4'
            elif (ynew==5):
                ynew='5'
            elif (ynew==6):
                ynew='6'
            elif (ynew==7):
                ynew='7'
            elif (ynew==8):
                ynew='8'
            elif (ynew==9):
                ynew='9'
            st.write("Model's prediction: "+ynew)
        else:
            st.write("Unknown gesture. Try again.")

    #check the file uploaded by the user
    if up_file is not None:
        img1=Image.open(up_file)
        st.image(img1,caption='Image uploaded by you')
        image_stuff(img1)
    else:
        print('Upload proper file.')
