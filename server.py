# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:41:26 2018

@author: kyler
"""
import os
import numpy as np
import json
from flask import Flask, jsonify, request
from keras.models import load_model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    #print('Begin API processing...')
    test_json = request.get_json()
    test = np.array(test_json)

    #print('Loading model...')
    filename = 'model.h5'
    model = load_model('models/' + filename)
    #print("The model has been successfully loaded...")

    #print('Classifying input')
    prediction = model.predict(test)
    prediction = prediction.tolist()
    #print('Predicted: ' , np.argmax(prediction))

    responses = jsonify(prediction=json.dumps(prediction))
    responses.status_code = 200

    return (responses)
    
app.run()