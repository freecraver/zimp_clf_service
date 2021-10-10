import json
import os
import tempfile
import unittest

from fasttext import load_model
from joblib import load

from app import app


class ClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.example_text = "What are Community Chest cards in Monopoly ?"
        self.example_texts = [self.example_text, "How many inches over six feet is the Venus de Milo ?"]

    def test_train(self):
        with open('res/train.csv', 'rb') as f:
            data = {'file': f, 'modelType': 'FASTTEXT', 'seed': 42}
            response = self.app.post('/train', data=data, follow_redirects=True,
                                     content_type='multipart/form-data')
            self.assertEqual(200, response.status_code)

    def test_predict(self):
        self.test_train()
        response = self.app.post('/predict', data=json.dumps({'text': self.example_text}),
                                 content_type='application/json')
        self.assertEqual(200, response.status_code)
        self.assertEqual('DESC', response.get_json().get('label'))

    def test_predict_proba(self):
        self.test_train()
        response = self.app.post('/predict_proba', data=json.dumps({'text': self.example_text, 'n': 3}),
                                 content_type='application/json')
        self.assertEqual(200, response.status_code)
        predictions = response.get_json()
        self.assertEqual(3, len(predictions))
        self.assertEqual('DESC', predictions[0].get('label'))
        self.assertTrue(predictions[0].get('probability') > 0.15)

    def test_predict_proba_multiple(self):
        self.test_train()
        response = self.app.post('/m/predict_proba', data=json.dumps({'texts': self.example_texts, 'n': 2}),
                                 content_type='application/json')
        self.assertEqual(200, response.status_code)
        predictions = response.get_json()
        self.assertEqual(2, len(predictions))
        self.assertEqual(self.example_text, predictions[0]['text'])
        self.assertEqual('DESC', predictions[0].get('labels')[0].get('label'))
        self.assertEqual('DESC', predictions[1].get('labels')[0].get('label'))
        self.assertTrue(predictions[0].get('labels')[0].get('probability') > 0.15)

    def test_download_model(self):
        self.test_train()
        response = self.app.get('/download')
        self.assertEqual(200, response.status_code)
        with tempfile.TemporaryDirectory() as td:
            f_name = os.path.join(td, 'model')
            with open(f_name, 'wb') as fh:
                fh.write(response.data)
            pipeline = load_model(f_name)
            self.assertIsNotNone(pipeline)
            self.assertEqual('__label__DESC', pipeline.predict([self.example_text])[0][0][0])


if __name__ == '__main__':
    unittest.main()
