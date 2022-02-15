from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from nlp.sklearn_model import BaseCountVectorizerModel


class DecisionTree(BaseCountVectorizerModel):
    def _init_classifier(self, **kwargs):
        return DecisionTreeClassifier(random_state=self.seed, min_samples_leaf=.01, **kwargs)

    def is_trained(self):
        return super().is_trained() and hasattr(self.text_clf.named_steps['clf'], 'tree_')


class RandomForest(BaseCountVectorizerModel):
    def _init_classifier(self, **kwargs):
        return RandomForestClassifier(
            random_state=self.seed,
            min_samples_leaf=.01,
            verbose=True,
            **kwargs)

    def is_trained(self):
        """
        random forest is done when all required attributes are created and the correct nr of estimators was created
        """
        if not super().is_trained():
            return False
        forest = self.text_clf.named_steps['clf']
        if not hasattr(forest, 'estimators_'):
            return False
        return len(forest.estimators_) >= forest.n_estimators
