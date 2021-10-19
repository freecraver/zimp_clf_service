from enum import Enum

from nlp.bert import Bert
from nlp.fasttext import FastText
from nlp.svm import SVM


class ModelType(Enum):
    SVM = 'SVM'
    FASTTEXT = 'FASTTEXT'
    BERT = 'BERT'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


# only works with single worker -> transfer trained model to key value store or object storage for scalability
class ClassificationProvider:

    __shared_state = None

    def __init__(self):
        if not ClassificationProvider.__shared_state:
            ClassificationProvider.__shared_state = self.__dict__
            self.model = None
        else:
            self.__dict__ = self.__shared_state

    def init_model(self, model_type: ModelType, seed=None):
        self.model = _init_model(model_type, seed)
        return self.get_model()

    def get_model(self):
        return self.model

    def has_model(self):
        return self.model is not None


def _init_model(model_type: ModelType, seed=None):
    if model_type == ModelType.SVM:
        return SVM(seed)
    elif model_type == ModelType.FASTTEXT:
        return FastText(seed)
    elif model_type == ModelType.BERT:
        return Bert(seed)

    raise EnvironmentError(f'model type {model_type} not supported')



