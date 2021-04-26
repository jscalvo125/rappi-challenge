from flask import Flask, request, jsonify
import pandas as pd
from titanic_ml.__main__ import PredictionWrapper
from logging.config import dictConfig
import psutil
import os
import sys
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

dirname = os.path.dirname(__file__)
model_filename = os.path.join(dirname, "../models/gradient_clf.pkl")
train_data_name = os.path.join(dirname, "../data/train.csv")
pred_wrapper = PredictionWrapper(model_filename, train_data_name)
app = Flask(__name__)
if len(sys.argv) < 2:
    app.logger.info("No model selected, loading default model....")
else:
    model_filename = sys.argv[1]
    app.logger.info("Loading ML model from {}".format(model_filename))
    pred_wrapper = PredictionWrapper(model_filename)
    app.logger.info("ML model loaded")


def get_profiling_info():
    app.logger.info("Virtual memory = {} %".format(psutil.virtual_memory().percent))
    app.logger.info(
        "Available memory = {} %".format(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total))
    app.logger.info("CPU usage = {} %".format(psutil.cpu_percent()))
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    app.logger.info('App''s memory use (in GB): {}'.format(memoryUse))


@app.route('/')
def hello_world():
    get_profiling_info()
    return 'Hi! This is an API for predicting Titanic survivals. Please read README.md to know how to use me!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        get_profiling_info()
        test_data = request.get_json()
        passengers = pd.DataFrame.from_dict(test_data)
        app.logger.info("Calling the prediction for passengers {}".format(passengers))
        passengers["predictions"] = pred_wrapper.predict(passengers)
        res = [{"passenger": passenger["Name"],
                "survived":passenger["predictions"]
                } for i, passenger in passengers.iterrows()]
        app.logger.info('predictions collected')
        get_profiling_info()
        return jsonify(status='200 OK', label=res)
    except Exception as e:
        app.logger.info("Failed to call the prediction method - {}".format(e))
        get_profiling_info()
        return jsonify(status='401', label=e)


get_profiling_info()
app.run(host='0.0.0.0', port=80)
