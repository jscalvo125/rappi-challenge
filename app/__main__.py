from flask import Flask, request, jsonify
import pandas as pd
from titanic_ml.__main__ import PredictionWrapper
import os
dirname = os.path.dirname(__file__)
model_filename = os.path.join(dirname, "../models/gradient_clf.pkl")
app = Flask(__name__)
app.logger.info("Loading ML model from {}".format(model_filename))
pred_wrapper = PredictionWrapper(model_filename)
app.logger.info("ML model loaded")

@app.route('/')
def hello_world():
    return 'Hi! This is an API for predicting Titanic survivals. Please read README.md to know how to use me!'


@app.route('/predict', methods=['POST'])
def predict():
    test_data = request.get_json()
    passengers = pd.DataFrame.from_dict(test_data)
    app.logger.info("Calling the prediction for passengers {}".format(passengers))
    passengers["predictions"] = pred_wrapper.predict(passengers)
    res = [{"passenger": passenger["Name"],
            "survived":passenger["predictions"]
            } for i, passenger in passengers.iterrows()]
    return jsonify(status='complete', label=res)


app.run(host='0.0.0.0', port=8080, debug=True)
