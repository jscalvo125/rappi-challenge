import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class PredictionWrapper:
    def __init__(self, model_name: str, train_data_path: str):
        try:
            self.model = pickle.load(open(model_name, "rb"))
            self.training_df = pd.read_csv(train_data_path)
            self.training_df['FamilySize'] = self.training_df['Parch'] + self.training_df['SibSp']
            self.scaler = MinMaxScaler()
            features = ['Age', 'Fare', 'FamilySize']
            self.scaler.fit(self.training_df[features])
            print("model found {}".format(model_name))
        except:
            raise "the model {} could not be loaded".format(model_name)

    def get_model(self):
        return self.model

    def pre_process(self, passengers: pd.DataFrame) -> pd.DataFrame:
        processed_df = passengers
        processed_df.Age = processed_df.Age.fillna(processed_df.Age.median())
        processed_df = processed_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

        processed_df['C'] = 0
        processed_df['Q'] = 0
        processed_df['S'] = 0
        embark_dummies_titanic = pd.get_dummies(processed_df['Embarked'])
        if 'C' in embark_dummies_titanic.keys():
            processed_df['C'] = embark_dummies_titanic['C']
        if 'Q' in embark_dummies_titanic.keys():
            processed_df['Q'] = embark_dummies_titanic['Q']
        if 'S' in embark_dummies_titanic.keys():
            processed_df['S'] = embark_dummies_titanic['S']

        processed_df['female'] = 0
        processed_df['male'] = 0
        sex_dummies_titanic = pd.get_dummies(processed_df['Sex'])
        if 'female' in sex_dummies_titanic.keys():
            processed_df['female'] = sex_dummies_titanic['female']
        if 'male' in sex_dummies_titanic.keys():
            processed_df['male'] = sex_dummies_titanic['male']

        processed_df['Class_1'] = 0
        processed_df['Class_2'] = 0
        processed_df['Class_3'] = 0
        pclass_dummies_titanic = pd.get_dummies(processed_df['Pclass'], prefix="Class")
        if 'Class_1' in pclass_dummies_titanic.keys():
            processed_df['Class_1'] = pclass_dummies_titanic['Class_1']
        if 'Class_2' in pclass_dummies_titanic.keys():
            processed_df['Class_2'] = pclass_dummies_titanic['Class_2']
        if 'Class_3' in pclass_dummies_titanic.keys():
            processed_df['Class_3'] = pclass_dummies_titanic['Class_3']

        processed_df = processed_df.drop(['Embarked', 'Sex', 'Pclass'], axis=1)
        processed_df['FamilySize'] = processed_df['Parch'] + processed_df['SibSp']
        processed_df = processed_df.drop(['Cabin', 'Parch', 'SibSp'], axis=1)
        features = ['Age', 'Fare', 'FamilySize']
        processed_df[features] = self.scaler.transform(processed_df[features])
        return processed_df

    def predict(self, passengers: pd.DataFrame):
        prediction = []
        if passengers is None:
            return prediction
        try:
            pre_processed_data = self.pre_process(passengers)
            prediction = self.model.predict(pre_processed_data)
            return prediction
        except Exception as e:
            raise "Error while calling the prediction process {}".format(e)
