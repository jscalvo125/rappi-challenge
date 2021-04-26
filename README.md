# Rappi ML Engineering Challenge

This repository includes a [jupyter notebook](./notebooks/rappi-ml-challenge.ipynb) that trains two models based on
Titanic dataset provided by Kaggle.

# Components

This repo contains the following packages/folders:

-  [app](./app) The Flask app that exposes and serves the ML prediction.
-  [data](./data) The training dataset used for training the ML models.
-  [models](./models) Contains all the trained models tested in the titanic dataset in pickle (binary) format.
-  [tests](./tests) Unit tests for the ML model and its pre-processing+prediction pipeline.
-  [titanic_ml](./titanic_ml) package to perform the pre-process+prediction

# How to install

To install, make sure you have Python==3.8.x and pip==latest in your environment.
Run: pip install -r requirements.txt to install the required dependencies.

# How to run
Type: python -m app to run the Flask app in your environment. The app points to 0.0.0.0:80/
Alternatively, run: python main.py to run the app.

You can load your own model(sklearn pickle file) by adding an extra argument to:
python -m app <your_model> if needed. The app loads an instance of GradientBoostingClassifier by default, which is the best performing model in our tests 

# How to use
Call the URI http://localhost/predict (change localhost to your IP address/host name) via POST with the following parameters:

## Headers
content-type: "Application/json"
## Body(an example)
[
    {
        "PassengerId": 1000,
        "Name": "Rose DeWiit Bukater",
        "Sex": "female",
        "Age": 17,
        "Cabin": "1A",
        "Pclass": "1",
        "Ticket": "AAA",
        "SibSp": 0,
        "Parch": 1,
        "Fare": 53.1,
        "Embarked": "S"
    }
]

## Response
{
    "label": [
        {
            "passenger": "Rose DeWiit Bukater",
            "survived": 1
        }
    ],
    "status": "200 OK"
}

# Deployment / Productionizing

If you want to deploy into a container, [Dockerfile](./Dockerfile) contains a deployment into a python 3.8 image with all the necessary components in it.
you can test it using [start.sh](./start.sh) if you run it in your linux env.