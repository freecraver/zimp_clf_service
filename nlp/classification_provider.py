from enum import Enum

from nlp.bert import Bert
from nlp.decision_tree import DecisionTree, RandomForest
from nlp.fasttext import FastText
from nlp.german_bert import GermanBert
from nlp.svm import SVM


class ModelType(Enum):
    SVM = 'SVM'
    FASTTEXT = 'FASTTEXT'
    BERT = 'BERT'
    GERMAN_BERT = 'GERMAN_BERT'
    DECISION_TREE = 'DECISION_TREE'
    RANDOM_FOREST = 'RANDOM_FOREST'

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

    def init_model(self, model_type: ModelType, seed=None, **kwargs):
        self.model = _init_model(model_type, seed, **kwargs)
        return self.get_model()

    def get_model(self):
        return self.model

    def has_model(self):
        return self.model is not None


def _init_model(model_type: ModelType, seed=None, **kwargs):
    if model_type == ModelType.SVM:
        return SVM(seed, **kwargs)
    elif model_type == ModelType.FASTTEXT:
        return FastText(seed, **kwargs)
    elif model_type == ModelType.BERT:
        return Bert(seed, **kwargs)
    elif model_type == ModelType.GERMAN_BERT:
        return GermanBert(seed, **kwargs)
    elif model_type == ModelType.DECISION_TREE:
        return DecisionTree(seed, **kwargs)
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForest(seed, **kwargs)

    raise EnvironmentError(f'model type {model_type} not supported')



