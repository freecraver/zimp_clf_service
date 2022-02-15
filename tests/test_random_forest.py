import unittest

from tests.base_sklearn_test import AbstractTest


class RandomForestClassificationTest(AbstractTest.BaseSklearnTest):

    def get_train_parameters(self):
        return {'modelType': 'RANDOM_FOREST', 'seed': 213}


if __name__ == '__main__':
    unittest.main()
