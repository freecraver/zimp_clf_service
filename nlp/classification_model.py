from abc import ABC, abstractmethod


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
