from sklearn.linear_model import SGDClassifier

from nlp.classification_model import Model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class SVM(Model):

    def __init__(self):
        self.text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(random_state=42))
        ])

    def train(self, X, y):
        self.text_clf.fit(X, y)



