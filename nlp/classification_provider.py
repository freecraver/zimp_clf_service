from config import MODEL_TYPE
from nlp.fasttext import FastText
from nlp.svm import SVM


class ClassificationProvider:
    __shared_state = None

    def __init__(self):
        if not ClassificationProvider.__shared_state:
            ClassificationProvider.__shared_state = self.__dict__
            self.model = self._init_model()
        else:
            self.__dict__ = self.__shared_state

    def get_model(self):
        return self.model

    def _init_model(self):
        if MODEL_TYPE == 'SVM':
            return SVM()
        elif MODEL_TYPE == 'FASTTEXT':
            return FastText()

        raise EnvironmentError('No valid MODEL_TYPE configured')
