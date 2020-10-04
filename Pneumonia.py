import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import os
import io
import base64
from werkzeug.utils import secure_filename
import cv2
app=Flask(__name__)
model=load_model('model_pneumonia.h5')
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)
    result=np.round(result)
    if result[0][0]==0.0:
        pred="The person has no Pneumonia" 
    elif result[0][0]==1.0:
        pred="The person has Pneumonia" 
    else:
        pred="may be you have gave other file instead of xray of a person"

    return pred


@app.route('/')
def home():
    return render_template('hackathon.html')

@app.route('/Pneumonia_Detection',methods=['POST','GET'])
def Pneumonia_Detection():
    '''
    For rendering results on HTML GUI
    '''
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
        '''result = preds
    test_image = request.form.values()
    test_image = image.img_to_array(test_image)
    test_image=test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    result=np.round(result)
    if result[0][0]==0.0:
        pred="The person has no Pneumonia" 
    elif result[0][0]==1.0:
        pred="The person has Pneumonia" 
    else:
        pred="may be you have gave other file instead of xray of a person"'''
    return render_template('hackathon.html', prediction_text='{}'.format(preds))

@app.route('/predict_api',methods=['POST']) 
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.stress_detecion([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)