import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from fasttext import load_model
from joblib import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from app import app
from nlp.bert import Bert


class ClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.example_text = "What are Community Chest cards in Monopoly ?"

    def test_train(self):
        with open('res/train.csv', 'rb') as f:
            data = {'file': f, 'modelType': 'BERT', 'seed': 42}
            response = self.app.post('/train', data=data, follow_redirects=True,
                                     content_type='multipart/form-data')
            self.assertEqual(200, response.status_code)

    def test_predict(self):
        self.test_train()
        response = self.app.post('/predict', data=json.dumps({'text': self.example_text}),
                                 content_type='application/json')
        self.assertEqual(200, response.status_code)
        self.assertEqual('HUM', response.get_json().get('label'))

    def test_predict_proba(self):
        self.test_train()
        response = self.app.post('/predict_proba', data=json.dumps({'text': self.example_text, 'n': 3}),
                                 content_type='application/json')
        self.assertEqual(200, response.status_code)
        predictions = response.get_json()
        self.assertEqual(3, len(predictions))
        self.assertEqual('HUM', predictions[0].get('label'))

    def test_download_model(self):
        self.test_train()
        response = self.app.get('/download')
        self.assertEqual(200, response.status_code)
        with tempfile.TemporaryDirectory() as td:
            f_name = os.path.join(td, 'model.zip')
            with open(f_name, 'wb') as fh:
                fh.write(response.data)
            shutil.unpack_archive(f_name, td)
            tokenizer = AutoTokenizer.from_pretrained(td)
            pipeline = AutoModelForSequenceClassification.from_pretrained(td)
            self.assertIsNotNone(pipeline)
            inputs = tokenizer(self.example_text, return_tensors="pt", padding="max_length", truncation=True)
            set_seed(23)
            with torch.no_grad():
                logits = pipeline(**inputs).logits.tolist()
            probs = Bert.softmax(logits)[0]
            ret_idx = (-1 * probs).argsort()[0]
            self.assertEqual(2, ret_idx)


if __name__ == '__main__':
    unittest.main()
