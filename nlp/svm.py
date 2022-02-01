import logging
import math
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.svm import SVC

from nlp.classification_model import Model, PREDICT_PROBA_N
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


ENABLE_PRE_PROCESSING = False


class SVM(Model):

    def __init__(self, seed=None):
        super(SVM, self).__init__(seed)

        vectorizer = CountVectorizer(lowercase=False)
        if not ENABLE_PRE_PROCESSING:
            vectorizer.tokenizer = lambda x: x.split()

        self.text_clf = Pipeline([
            ('vect', vectorizer),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC(random_state=self.seed, verbose=True))
        ], verbose=True)

    def train(self, X, y):
        max_iter = max(100, math.floor(math.pow(10, 9)/len(X)))  # set max_iter to 10^9/n
        logging.debug(f'Starting with training (max_iter={max_iter})')
        self.text_clf.set_params(clf__max_iter=max_iter)
        self.text_clf.fit(X, y.astype(str))

    def is_trained(self):
        return hasattr(self.text_clf, 'classes_') and hasattr(self.text_clf.named_steps['clf'], 'shape_fit_')

    def get_dumped_model_path(self):
        tmp_file = Path('svm.joblib')
        dump(self.text_clf, tmp_file)
        return tmp_file.resolve()

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        """
        note that SVM does not return probabilities, but softmaxed outputs of the decision function,
        this decision was made as the platt scaling for probability estimates works quite bad for small datasets
         https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
        :param X: one input text
        :param n: number of labels to return, in case less target labels were trained the number of returned labels
            might be smaller
        :return: iterable with entries of shape [class_label, decision_val], [str, float], sorted descending by
        decision value
        """
        dv = self.text_clf.decision_function(X)

        if self.text_clf.classes_.size == 2:
            # SVM with two classes has a decision function which returns a scalar value (threshold at 0)
            # map to multi-class decision values
            dv = np.stack([0 - dv, dv], axis=1)

        ps = self.softmax(dv)
        ret_idx = (-1*ps).argsort()[:, :n]
        ps_ret = ps[np.repeat(np.arange(len(ps)),n),ret_idx.flatten()].reshape(len(ps), n)
        return np.stack([self.text_clf.classes_[ret_idx], ps_ret], axis=2)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
