import json
import os
import tempfile
import unittest

from joblib import load

from app import app


class ClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.example_text = "How many Community Chest cards are there in Monopoly ?"

    def test_train(self):
        with open('res/train.csv', 'rb') as f:
            data = {'file': f}
            response = self.app.post('/train', data=data, follow_redirects=True,
                                     content_type='multipart/form-data')
            self.assertEqual(200, response.status_code)

    def test_predict(self):
        self.test_train()
        response = self.app.post('/predict', data=json.dumps({'text': self.example_text}),
                                 content_type='application/json')
        self.assertEqual(200, response.status_code)
        self.assertEqual('NUM', response.get_json().get('label'))

    def test_predict_proba(self):
        self.test_train()
        response = self.app.post('/predict_proba', data=json.dumps({'text': self.example_text, 'n': 3}),
                                 content_type='application/json')
        self.assertEqual(200, response.status_code)
        predictions = response.get_json()
        self.assertEqual(3, len(predictions))
        self.assertEqual('NUM', predictions[0].get('label'))

    def test_download_model(self):
        self.test_train()
        response = self.app.get('/download')
        self.assertEqual(200, response.status_code)
        with tempfile.TemporaryDirectory() as td:
            f_name = os.path.join(td, 'model')
            with open(f_name, 'wb') as fh:
                fh.write(response.data)
            pipeline = load(f_name)  # assumes SVM model
            self.assertIsNotNone(pipeline)
            self.assertEqual('NUM', pipeline.predict([self.example_text])[0])


if __name__ == '__main__':
    unittest.main()
