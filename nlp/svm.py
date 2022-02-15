import logging
import math
import numpy as np
from sklearn.svm import SVC

from nlp.classification_model import PREDICT_PROBA_N
from nlp.sklearn_model import BaseCountVectorizerModel


class SVM(BaseCountVectorizerModel):

    def _init_classifier(self, **kwargs):
        return SVC(random_state=self.seed, verbose=True, **kwargs)

    def _pre_train(self, X, y):
        max_iter = max(100, math.floor(math.pow(10, 9) / len(X)))  # set max_iter to 10^9/n
        logging.debug(f'Setting max_iter to {max_iter}')
        self.text_clf.set_params(clf__max_iter=max_iter)

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
        return self.transform_prediction_output(ps, n, self.text_clf.classes_)

    def is_trained(self):
        return super().is_trained() and hasattr(self.text_clf.named_steps['clf'], 'shape_fit_')


