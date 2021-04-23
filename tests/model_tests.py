from titanic_ml.__main__ import PredictionWrapper
import pandas as pd
import unittest
import os
dirname = os.path.dirname(__file__)
model_filename = os.path.join(dirname, "../models/gradient_clf.pkl")


class TestTitanicModel(unittest.TestCase):

    def test_predict_empty_df(self):
        pred_wrapper = PredictionWrapper(model_filename)
        predictions = pred_wrapper.predict(None)
        self.assertEqual(len(predictions), 0)

    def test_predict(self):
        pred_wrapper = PredictionWrapper(model_filename)
        my_df = pd.read_csv(os.path.join(dirname, "../data/train.csv"))
        test_df = my_df[0:2]
        test_df = test_df.drop(["Survived"], axis=1)
        predictions = pred_wrapper.predict(test_df)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0], 0)
        self.assertEqual(predictions[1], 1)


if __name__ == '__main__':
    unittest.main()