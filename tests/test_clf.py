import unittest
from app import app


class ClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_train(self):
        with open('res/train.csv', 'rb') as f:
            data = {'file': f}
            response = self.app.post('/train', data=data, follow_redirects=True,
                                     content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
