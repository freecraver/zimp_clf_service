import numpy as np

from sklearn.svm import SVC
from nlp.classification_model import Model, PREDICT_PROBA_N
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class SVM(Model):

    def __init__(self):
        self.text_clf = Pipeline([
            ('vect', CountVectorizer(lowercase=False)),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC(random_state=42, probability=True))
        ])

    def train(self, X, y):
        self.text_clf.fit(X, y)

    def is_trained(self):
        return hasattr(self.text_clf, 'classes_')

    def predict(self, X):
        return self.text_clf.predict([X])[0]

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        """
        note that probability estimates might not be perfectly calibrated for SVC, and top class label might not be the
        same as for predict
         https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
        :param X: one input text
        :param n: number of labels to return, in case less target labels were trained the number of returned labels
            might be smaller
        :return: iterable with entries of shape [class_label, probability], [str, float], sorted descending by
        probability
        """
        p = self.text_clf.predict_proba([X])
        ret_idx = (-1*p).argsort()[:n]
        return np.stack([self.text_clf.classes_[ret_idx], p.flatten()[ret_idx]], axis=2)[0]




