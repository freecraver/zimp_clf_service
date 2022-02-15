import unittest

from tests.base_sklearn_test import AbstractTest


class DecisionTreeClassificationTest(AbstractTest.BaseSklearnTest):

    def get_train_parameters(self):
        return {'modelType': 'DECISION_TREE', 'seed': 213}


if __name__ == '__main__':
    unittest.main()
