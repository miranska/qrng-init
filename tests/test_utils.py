import unittest
import numpy as np

from utils import split_data


class TestSplitData(unittest.TestCase):

    def setUp(self):
        self.x_train = np.random.rand(100, 64)  # 100 samples, 64 features each
        self.y_train = np.random.rand(
            100, 10
        )  # 100 samples, 10 target values each

    def test_split_data_full_training(self):
        (x_train, y_train), (x_val, y_val) = split_data(
            self.x_train, self.y_train, 100
        )
        self.assertEqual(x_train.shape, (100, 64))
        self.assertEqual(y_train.shape, (100, 10))
        self.assertEqual(x_val.shape, (0, 64))
        self.assertEqual(y_val.shape, (0, 10))

    def test_split_data_partial_training(self):
        (x_train, y_train), (x_val, y_val) = split_data(
            self.x_train, self.y_train, 80
        )
        self.assertEqual(x_train.shape, (80, 64))
        self.assertEqual(y_train.shape, (80, 10))
        self.assertEqual(x_val.shape, (20, 64))
        self.assertEqual(y_val.shape, (20, 10))

    def test_split_data_minimal_training(self):
        (x_train, y_train), (x_val, y_val) = split_data(
            self.x_train, self.y_train, 1
        )
        self.assertEqual(x_train.shape, (1, 64))
        self.assertEqual(y_train.shape, (1, 10))
        self.assertEqual(x_val.shape, (99, 64))
        self.assertEqual(y_val.shape, (99, 10))

    def test_split_data_zero_training(self):
        (x_train, y_train), (x_val, y_val) = split_data(
            self.x_train, self.y_train, 0
        )
        self.assertEqual(x_train.shape, (0, 64))
        self.assertEqual(y_train.shape, (0, 10))
        self.assertEqual(x_val.shape, (100, 64))
        self.assertEqual(y_val.shape, (100, 10))


if __name__ == "__main__":
    unittest.main()
