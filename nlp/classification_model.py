import csv
import logging
from abc import ABC, abstractmethod
from threading import Thread
from pathlib import Path

import numpy as np

PREDICT_PROBA_N = 10


class Model(ABC):
    seed = None

    def __init__(self, seed=None):
        self.seed = seed

    @abstractmethod
    def train(self, X, y):
        """
        trains a text classification model using the supplied data
        any kind of hyperparameter config, etc. should be loaded in the respective child class
        to keep this interface as minimalistic as possible
        :param X: iterable of training texts
        :param y: iterable of training labels
        :return: true, if successful
        """
        raise NotImplementedError

    @abstractmethod
    def is_trained(self):
        """

        :return: true if model has been trained previously
        """
        return False

    @abstractmethod
    def predict_proba(self, X, n=PREDICT_PROBA_N):
        """
        returns the prediction probabilities for the top n target labels with prediction probability > 0
        :param X: array of text examples [str]
        :param n: number of labels to return, in case less target labels were trained the number of returned labels
            might be smaller
        :return: iterable of iterables with entries of shape [class_label, probability], [str, float], sorted descending by
        probability
        """
        raise NotImplementedError

    @abstractmethod
    def get_dumped_model_path(self):
        """
        persists a trained model to a temp file, representation depends on actual model
        :return: path to persisted temp model
        """
        raise NotImplementedError

    def get_prediction_batch_size(self) -> int:
        """
        :return: number of texts to be predicted per batch
        """
        return 128

    def predict(self, X):
        """
        uses the trained model to predict the most likely class label
        :param X: array of text examples [str]
        :return: array of predicted class labels [str]
        """
        res = self.predict_proba(X, 1)
        return [p[0, 0] for p in res]

    def _predict_file_sync(self, texts, result_id):
        batch_size = self.get_prediction_batch_size()
        file_name = f"results_{result_id}.csv"

        # delete any previous prediction files
        for p in Path('.').glob('results_*.csv'):
            p.unlink()

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('text,prediction,certainty\n')

        text_cnt = len(texts)

        for idx in range(0, text_cnt, batch_size):
            if idx % (batch_size*100) == 0:
                logging.info(f"Working on idx {idx} of {text_cnt}")
            cur_texts = texts[idx:idx + batch_size]
            predictions = self.predict_proba(cur_texts, 1)
            self._append_to_prediction_file(cur_texts, predictions, file_name)

    @staticmethod
    def _append_to_prediction_file(texts, prediction_tuples, file_name):
        with open(file_name, "a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'prediction', 'certainty'])
            for text, pred in zip(texts, prediction_tuples):
                top_pred = pred[0]
                writer.writerow({'text': text, 'prediction': top_pred[0], 'certainty': top_pred[1]})

    def predict_file(self, texts, result_id):
        """
        performs async batched prediction and writes results to a csv file after each nth batch
        :param texts: iterable of texts to predict
        :param result_id: id of result to be tracked
        :return: true
        """
        Thread(target=self._predict_file_sync, args=(texts, result_id)).start()

    def train_async(self, X, y):
        """
        trains a text classification model using the supplied data asynchronously
        check if training finished using is_trained
        @see train
        :param X: iterable of training texts
        :param y: iterable of training labels
        :return: nothing
        """
        Thread(target=self.train, args=(X, y)).start()

    @staticmethod
    def transform_prediction_output(ps: np.array, n: int, class_labels: np.array):
        """
        transform model prediction to expected classifier format
        :param ps: 2D-array of class-label probabilities (or estimates) [num_texts x num_class_labels]
        :param n: number of top results to be returned per prediction, interval of [1, num_class_labels]
        :param class_labels: array of class labels in the order of self.predict_proba indices
        :return: 3D array with entries of shape [num_texts, n, 2], whereas third dim is of form
         (class_label, probability), i.e. [str, float], sorted descending by probability
        """
        ret_idx = (-1 * ps).argsort()[:, :n]  # identify top n indices per example
        # extract top n probability values in descending order at reshape to  [num_texts x num_class_labels]
        ps_ret = ps[np.repeat(np.arange(len(ps)), n), ret_idx.flatten()].reshape(len(ps), n)
        # join class labels with probability values accross third dimension
        return np.stack([class_labels[ret_idx], ps_ret], axis=2)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
