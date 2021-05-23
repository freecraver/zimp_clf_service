from nlp.svm import SVM


class ClassificationProvider:
    __shared_state = None

    def __init__(self):
        if not ClassificationProvider.__shared_state:
            ClassificationProvider.__shared_state = self.__dict__
            self.model = SVM()
        else:
            self.__dict__ = self.__shared_state

    def get_model(self):
        return self.model
