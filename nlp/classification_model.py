from abc import ABC, abstractmethod

PREDICT_PROBA_N=10

class Model(ABC):

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
        :param X: one text example (str)
        :param n: number of labels to return, in case less target labels were trained the number of returned labels
            might be smaller
        :return: iterable with entries of shape [class_label, probability], [str, float], sorted descending by
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

    def predict(self, X):
        """
        uses the trained model to predict the most likely class label
        :param X: one text example (str)
        :return: predicted class label (str)
        """
        return self.predict_proba(X, 1)[0, 0]
