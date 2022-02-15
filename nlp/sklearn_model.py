import logging
from abc import ABC, abstractmethod
from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from nlp.classification_model import Model, PREDICT_PROBA_N

ENABLE_PRE_PROCESSING = False


class BaseCountVectorizerModel(Model, ABC):

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed)

        vectorizer = CountVectorizer(lowercase=False)
        if not ENABLE_PRE_PROCESSING:
            vectorizer.tokenizer = str.split

        self.text_clf = Pipeline([
            ('vect', vectorizer),
            ('tfidf', TfidfTransformer()),
            ('clf', self._init_classifier(**kwargs))
        ], verbose=True)
        self._is_trained = False

    @abstractmethod
    def _init_classifier(self, **kwargs):
        """
        :return: instance of sklearn classifier which uses tf-idf transformed vectors
        """
        pass

    def _pre_train(self, X, y):
        """
        executed before actual training takes place
        might be used to init clf-specific paramaters
        :param X:
        :param y:
        :return:
        """
        pass

    def train(self, X, y):
        self._is_trained = False
        self._pre_train(X, y)
        logging.debug('Starting with training')
        self.text_clf.fit(X, y.astype(str))
        self._is_trained = True

    def is_trained(self):
        return self._is_trained and hasattr(self.text_clf, 'classes_')

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        ps = self.text_clf.predict_proba(X)
        return self.transform_prediction_output(ps, n, self.text_clf.classes_)

    def get_dumped_model_path(self):
        tmp_file = Path('sklearn.joblib')
        dump(self.text_clf, tmp_file)
        return tmp_file.resolve()