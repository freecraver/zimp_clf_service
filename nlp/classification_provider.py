from nlp.svm import SVM


class ClassificationProvider:
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
        self.model = SVM()

    def get_model(self):
        return self.model
