
from PIL import Image
import cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt
import math
from tensorflow.keras import models, layers
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import pickle
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
import re
import glob
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import numpy as np
from flask import Flask, redirect, render_template, request, url_for
app = Flask(__name__)



@app.route('/')
def Diabetic_retinopathy():
    return render_template("index.html")


model_path ='leaf_disease_weights.h5'

model = load_model(model_path)
class_names = np.array(
    ['Apple_Apple_scab',
'Apple_Black_rot',
'Apple_Cedar_apple_rust',
'Apple_healthy',
'Background_without_leaves',
'Blueberry_healthy',
'Cherry_healthy',
'Cherry_Powdery_mildew',
'Com_Common_rust',
'Corn_healthy',
'Corn_Northern_Leaf_Blight',
'Grape_Black_rot',
'Grape_healthy',
'Peach_Bacterial_spot',
'Peach_healthy',
'Pepper_bell_healthy',
'Potato_Early_blight',
'Potato_healthy',
'Potato_Late_blight',
'Raspberry_healthy',
'Soybean_healthy',
'Squash_Powdery_mildew',
'Strawberry_healthy',
'Strawberry_Leaf_scorch',
'Tomato_Bacterial_spot',
'Tomato_Early_blight',
'Tomato_healthy',
'Tomato_Late_blight',
'Tomato_Leaf_Mold',
'Tomato_Septoria_leaf_spot',
'Tomato_Target_Spot'])


@app.route('/image_predict', methods=['GET', 'POST'])
def image_predict():
    if (request.method == 'POST'):
        f = request.files['pic']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # # Make prediction
        # image=imread(file_path)
        # predicted_class, confidence,val  = predict1(model,image)

        # img = im.fromarray(imv age)

        # Load the image using PIL library
        # image = Image.open(f)
        # image = image.resize((512,512))  # Resize the image to 512x512 pixels
        # # Convert grayscale to RGB format
        # image = image.convert("RGB")
        # img = np.array(image).reshape(1,512, 512, 3)  # Reshape to (batch_size, height, width, channels)
        # img = img / 255

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        # Resize to match the model's input size
        img = cv2.resize(img, (512, 512))
        # Reshape to (batch_size, height, width, channels)
        img = np.array(img).reshape(1, 512, 512, 3)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        result = model.predict(img)

        predicted_class_index = np.argmax(result)
        predicted_class = class_names[predicted_class_index]
        print("Predicted class:", predicted_class)

        s = str(result[0])

        res = 'The predicted stage is \'' + predicted_class + \
            "\'" + str(predicted_class_index) + " " + s
    return render_template("open2.html", n=res)


if __name__ == "__main__":
    app.run(debug=True)
