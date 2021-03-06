import unittest

from tests.base_sklearn_test import AbstractTest


class SvmClassificationTest(AbstractTest.BaseSklearnTest):

    def get_train_parameters(self):
        return {'modelType': 'SVM', 'seed': 213}


if __name__ == '__main__':
    unittest.main()
