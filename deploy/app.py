from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='D:/project/deploy/Xception_300d/best.h5'

# Load your trained model
model = load_model(MODEL_PATH)


print("Model loaded http://127.0.0.1:5000/")

def model_predict(img_path, model):
    CATEGORIES = ['Black-grass (Weed)',
          'Charlock (Weed)' ,
          'Cleavers (Weed)',
          'Common Chickweed (Weed)',
          'Common wheat (Crop)',
          'Fat Hen (Weed)',
          'Loose Silky-bent (Weed)',
          'Maize (Crop)',
          'Scentless Mayweed (Weed)',
          'Shepherds Purse (Weed)',
          'Small-flowered Cranesbill (Weed)',
          'Sugar beet (Crop)']
    img = image.load_img(img_path, target_size=(300,300))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)
    x=preprocess_input(x)
    prediction=model.predict(x)
    preds = np.argmax(prediction)
    preds=CATEGORIES[preds]
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)