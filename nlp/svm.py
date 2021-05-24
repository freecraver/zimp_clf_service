import numpy as np
from joblib import dump

from sklearn.svm import SVC
from nlp.classification_model import Model, PREDICT_PROBA_N
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class SVM(Model):

    def __init__(self):
        self.text_clf = Pipeline([
            ('vect', CountVectorizer(lowercase=False)),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC(random_state=42))
        ])

    def train(self, X, y):
        self.text_clf.fit(X, y)

    def is_trained(self):
        return hasattr(self.text_clf, 'classes_')

    def get_dumped_model_path(self):
        tmp_file = 'svm.joblib'
        dump(self.text_clf, tmp_file)
        return tmp_file

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        """
        note that SVM does not return probabilities, but outputs of the decision function,
        this decision was made as the platt scaling for probability estimates works quite bad for small datasets
         https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
        :param X: one input text
        :param n: number of labels to return, in case less target labels were trained the number of returned labels
            might be smaller
        :return: iterable with entries of shape [class_label, decision_val], [str, float], sorted descending by
        decision value
        """
        dv = self.text_clf.decision_function([X])
        ret_idx = (-1*dv).argsort()[:, :n]
        return np.stack([self.text_clf.classes_[ret_idx], dv.flatten()[ret_idx]], axis=2)[0]




