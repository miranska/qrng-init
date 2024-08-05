import unittest
import numpy as np
from unittest.mock import patch

import pytest

from custom_initializers import (
    _get_seed_value,
    RandomNormalQR,
    RandomUniformQR,
    TruncatedNormalQR,
    GlorotUniformQR,
    GlorotNormalQR,
    HeNormalQR,
    HeUniformQR,
    LecunUniformQR,
    LecunNormalQR,
    OrthogonalInitializerQR,
)


class TestGetSeedValueFunction(unittest.TestCase):
    @patch("custom_initializers.get_suggested_seed_value")
    def test_reinit_true(self, mock_get_seed):
        mock_get_seed.return_value = 42
        initializer_name = "initializer"

        result = _get_seed_value(initializer_name)

        self.assertEqual(42, result)

    def test_reinit_false_seed_provided(self):
        initializer_name_1 = "initializer1"
        initializer_name_2 = "initializer2"
        self.assertEqual(_get_seed_value(initializer_name_1), 1)
        self.assertEqual(_get_seed_value(initializer_name_2), 1)
        self.assertEqual(_get_seed_value(initializer_name_1), 2)
        self.assertEqual(_get_seed_value(initializer_name_2), 2)


class TestRandomNormalQR:

    def test_distribution_parameters(self):
        initializer = RandomNormalQR(mean=0.0, stddev=1.0)
        values = initializer(shape=(1000,))
        # Assert mean and stddev are within expected ranges using
        # a simple statistical test
        assert np.isclose(np.mean(values), 0.0, atol=0.1)
        assert np.isclose(np.std(values), 1.0, atol=0.1)


class TestRandomUniformQR:

    def test_distribution_parameters(self):
        initializer = RandomUniformQR(minval=0.0, maxval=1.0)
        values = initializer(shape=(1000,))
        # Assert mean and stddev are within expected ranges using
        # a simple statistical test
        assert np.isclose(np.mean(values), 0.5, atol=0.1)
        assert np.isclose(np.std(values), np.sqrt(1 / 12), atol=0.1)


class TestTruncatedNormalQR:

    def test_distribution_parameters(self):
        initializer = TruncatedNormalQR(mean=0.0, stddev=1.0)
        values = initializer(shape=(1000,))
        # Assert mean and stddev are within expected ranges using
        # a simple statistical test
        assert np.isclose(np.mean(values), 0, atol=0.1)
        assert np.isclose(np.std(values), 0.8796, atol=0.1)


@pytest.fixture(
    params=[
        RandomNormalQR,
        RandomUniformQR,
        TruncatedNormalQR,
        GlorotNormalQR,
        GlorotUniformQR,
        HeNormalQR,
        HeUniformQR,
        LecunNormalQR,
        LecunUniformQR,
    ]
)
def model_class(request):
    return request.param


class TestCoreFunctionalityOfInitializer:
    def test_output_shape(self, model_class):
        initializer = model_class()
        shape = (10, 10)
        values = initializer(shape=shape)
        assert values.shape == shape

    def test_data_type_handling(self, model_class):
        initializer = model_class()
        dtype = np.float32
        shape = (10, 10)
        values = initializer(shape=shape, dtype=dtype)
        assert values.dtype == dtype

    def test_variability_between_objects(self, model_class):
        initializer1 = model_class()
        initializer2 = model_class()
        values1 = initializer1(shape=(100,))
        values2 = initializer2(shape=(100,))
        # Ensure that the values are not exactly the same
        assert not np.array_equal(values1, values2)

    def test_variability_between_invocations(self, model_class):
        initializer = model_class()
        values1 = initializer(shape=(100,))
        values2 = initializer(shape=(100,))
        # Ensure that the values are not exactly the same
        assert not np.array_equal(values1, values2)


class TestOrthogonalInitializerQR:

    def test_output_shape(self):
        initializer = OrthogonalInitializerQR()
        shape = (10, 10)
        values = initializer(shape=shape)
        assert values.shape == shape

    def test_data_type_handling(self):
        initializer = OrthogonalInitializerQR()
        dtype = np.float32
        shape = (10, 10)
        values = initializer(shape=shape, dtype=dtype)
        assert values.dtype == dtype

    def test_variability_between_objects(self):
        initializer1 = OrthogonalInitializerQR()
        initializer2 = OrthogonalInitializerQR()
        values1 = initializer1(shape=(100, 1))
        values2 = initializer2(shape=(100, 1))
        # Ensure that the values are not exactly the same
        assert not np.array_equal(values1, values2)

    def test_variability_between_invocations(self):
        initializer = OrthogonalInitializerQR()
        values1 = initializer(shape=(100, 1))
        values2 = initializer(shape=(100, 1))
        # Ensure that the values are not exactly the same
        assert not np.array_equal(values1, values2)


if __name__ == "__main__":
    unittest.main()
