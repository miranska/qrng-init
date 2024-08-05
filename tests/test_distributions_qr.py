import unittest

import keras
from keras import ops
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import numpy as np
import scipy
from scipy.stats import norm


from distributions_qr import (
    get_sobol_sequence_for_specific_dimension,
    random_uniform,
    random_normal,
    truncated_normal,
)


class TestGetSobolSequenceForSpecificDimension(unittest.TestCase):
    def test_dim_0_rows_1(self):
        expected_values = np.array([0.5])
        np.testing.assert_equal(
            expected_values,
            get_sobol_sequence_for_specific_dimension(dim_id=1, num_results=1),
        )

    def test_dim_0_rows_5(self):
        expected_values = np.array([0.5, 0.75, 0.25, 0.375, 0.875])
        np.testing.assert_equal(
            expected_values,
            get_sobol_sequence_for_specific_dimension(dim_id=1, num_results=5),
        )

    def test_dim_2_rows_4(self):
        expected_values = np.array([0.5, 0.25, 0.75, 0.625])
        np.testing.assert_equal(
            expected_values,
            get_sobol_sequence_for_specific_dimension(dim_id=3, num_results=4),
        )

    def test_different_data_types(self):
        float32_values = get_sobol_sequence_for_specific_dimension(
            dim_id=2, num_results=4, dtype="float32"
        )
        float64_values = get_sobol_sequence_for_specific_dimension(
            dim_id=2, num_results=4, dtype="float64"
        )

        # different data types
        with self.assertRaises(InvalidArgumentError):
            np.testing.assert_equal(float32_values, float64_values)

        # same values
        np.testing.assert_array_equal(
            float32_values, ops.cast(float64_values, dtype="float32")
        )


class TestRandomUniform(unittest.TestCase):
    def test_shapes(self):
        actual_values = random_uniform([3, 2], seed=3, dtype="float32")
        self.assertEqual([3, 2], actual_values.shape)

        actual_values = random_uniform([3], seed=3, dtype="float32")
        self.assertEqual([3], actual_values.shape)

        actual_values = random_uniform([3, 4, 5], seed=3, dtype="float32")
        self.assertEqual([3, 4, 5], actual_values.shape)

    def test_seed_change(self):
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                random_uniform([5], seed=2), random_uniform([5], seed=1)
            )
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                random_uniform([5], seed=6), random_uniform([5], seed=10)
            )

    def test_random_uniform_seed_0(self):
        expected_values = np.array([[0.5, 0.75, 0.25], [0.375, 0.875, 0.625]])
        np.testing.assert_equal(
            expected_values, random_uniform([2, 3], seed=1)
        )

    def test_random_uniform_seed_2(self):
        expected_values = np.array(
            [[0.5, 0.25], [0.75, 0.625], [0.125, 0.875]]
        )
        np.testing.assert_equal(
            expected_values, random_uniform([3, 2], seed=3)
        )

    def test_random_uniform_custom_range(self):
        expected_values = np.array([[0.0, -5.0], [5.0, 2.5], [-7.5, 7.5]])
        np.testing.assert_equal(
            expected_values,
            random_uniform([3, 2], seed=3, minval=-10, maxval=10),
        )

    def test_different_data_types(self):
        float32_values = random_uniform([3, 2], seed=3, dtype="float32")
        float64_values = random_uniform([3, 2], seed=3, dtype="float64")

        # different data types
        with self.assertRaises(InvalidArgumentError):
            np.testing.assert_equal(float32_values, float64_values)

        # same values
        np.testing.assert_array_equal(
            float32_values, ops.cast(float64_values, "float32")
        )

    def compare_four_moments_of_distributions(
        self, actual_values, min_value, max_value
    ):
        """
        Compare the first four moments (mean, vairance, skewness, kurtosis) of
        the distribution against theoretical values.

        :param actual_values: actual data
        :param min_value: minimum value of the distribution
        :param max_value: maximum value of the distribution

        :return: None
        """
        # the expected values are taken from
        # https://mathworld.wolfram.com/UniformDistribution.html
        expected_mean = 0.5 * (min_value + max_value)
        expected_variance = 1.0 / 12.0 * (max_value - min_value) ** 2
        expected_skewness = 0
        expected_kurtosis = -6.0 / 5.0

        self.assertAlmostEqual(expected_mean, np.mean(actual_values), places=4)
        self.assertAlmostEqual(
            expected_variance, np.var(actual_values), places=4
        )
        self.assertAlmostEqual(
            expected_skewness, scipy.stats.skew(actual_values), places=4
        )
        self.assertAlmostEqual(
            expected_kurtosis, scipy.stats.kurtosis(actual_values), places=4
        )

    def test_compare_in_distribution_standard_scale(self):
        sample_cnt = 100000
        actual_values = random_uniform([sample_cnt], seed=1).numpy()
        self.compare_four_moments_of_distributions(
            actual_values, min_value=0.0, max_value=1.0
        )

    def test_compare_in_distribution_custom_scale(self):
        sample_cnt = 100000
        min_val = -0.5
        max_val = 0.7
        actual_values = random_uniform(
            [sample_cnt], minval=min_val, maxval=max_val, seed=1
        ).numpy()
        self.compare_four_moments_of_distributions(
            actual_values, min_value=min_val, max_value=max_val
        )

    def test_no_seed(self):
        with self.assertRaises(ValueError):
            random_uniform([1])


class TestRandomNormal(unittest.TestCase):
    def test_shapes(self):
        actual_values = random_normal([3, 2], seed=3, dtype="float32")
        self.assertEqual([3, 2], actual_values.shape)

        actual_values = random_normal([3], seed=3, dtype="float32")
        self.assertEqual([3], actual_values.shape)

        actual_values = random_normal([3, 4, 5], seed=3, dtype="float32")
        self.assertEqual([3, 4, 5], actual_values.shape)

    def test_seed_change(self):
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                random_normal([5], seed=1), random_normal([5], seed=2)
            )
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                random_normal([5], seed=6), random_normal([5], seed=10)
            )

    def test_different_data_types(self):
        float32_values = random_normal([3, 2], seed=3, dtype="float32")
        float64_values = random_normal([3, 2], seed=3, dtype="float64")

        # different data types
        with self.assertRaises(InvalidArgumentError):
            np.testing.assert_equal(float32_values, float64_values)

        # values are close but not identical (due to numerical errors)
        np.testing.assert_array_almost_equal(
            float32_values, ops.cast(float64_values, "float32")
        )

    def compare_four_moments_of_distributions(
        self, actual_values, mean, stddev
    ):
        """
        Compare the first four moments (mean, variance, skewness, kurtosis) of
        the distribution against theoretical values.

        :param actual_values: actual data
        :param mean:  expected mean
        :param stddev: expected standard deviation

        :return: None
        """
        # the expected values are taken from
        # https://mathworld.wolfram.com/NormalDistribution.html
        expected_mean = mean
        expected_variance = stddev * stddev
        expected_skewness = 0
        expected_kurtosis = 0

        self.assertAlmostEqual(expected_mean, np.mean(actual_values), places=2)
        self.assertAlmostEqual(
            expected_variance, np.var(actual_values), places=1
        )
        self.assertAlmostEqual(
            expected_skewness, scipy.stats.skew(actual_values), places=1
        )
        self.assertAlmostEqual(
            expected_kurtosis, scipy.stats.kurtosis(actual_values), places=1
        )

    def test_compare_in_distribution_standard_scale(self):
        sample_cnt = 10000
        actual_values = random_normal([sample_cnt], seed=1).numpy()
        self.compare_four_moments_of_distributions(
            actual_values, mean=0.0, stddev=1.0
        )

    def test_compare_in_distribution_custom_scale(self):
        sample_cnt = 10000
        mean = 5.0
        stddev = 0.7
        actual_values = random_normal(
            [sample_cnt], mean=mean, stddev=stddev, seed=1
        ).numpy()
        self.compare_four_moments_of_distributions(
            actual_values, mean=mean, stddev=stddev
        )

    def test_no_seed(self):
        with self.assertRaises(ValueError):
            random_normal([1])


class TestTruncatedNormal(unittest.TestCase):
    def test_shapes(self):
        actual_values = truncated_normal([3, 2], seed=3, dtype="float32")
        self.assertEqual([3, 2], actual_values.shape)

        actual_values = truncated_normal([3], seed=3, dtype="float32")
        self.assertEqual([3], actual_values.shape)

        actual_values = truncated_normal([3, 4, 5], seed=3, dtype="float32")
        self.assertEqual([3, 4, 5], actual_values.shape)

    def test_seed_change(self):
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                truncated_normal([5], seed=1), truncated_normal([5], seed=2)
            )
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                truncated_normal([5], seed=6), truncated_normal([5], seed=10)
            )

    def test_different_data_types(self):
        float32_values = truncated_normal([3, 2], seed=3, dtype="float32")
        float64_values = truncated_normal([3, 2], seed=3, dtype="float64")

        # different data types
        with self.assertRaises(InvalidArgumentError):
            np.testing.assert_equal(float32_values, float64_values)

        # values are close but not identical (due to numerical errors)
        np.testing.assert_array_almost_equal(
            float32_values, ops.cast(float64_values, "float32")
        )

    def compare_three_moments_of_distributions(
        self, actual_values, mean, stddev, min_value, max_value
    ):
        """
        Compare the first four moments (mean, variance, skewness) of the
        distribution against theoretical values.
        Kurtosis is not readily available, skipping it.

        :param actual_values: actual data
        :param mean:  expected mean
        :param stddev: expected standard deviation
        :param min_value: minimum value of the distribution
        :param max_value: maximum value of the distribution

        :return: None
        """
        # The expected values are taken from
        # https://en.wikipedia.org/wiki/Truncated_normal_distribution
        # We are following the notation from the same page
        alpha = (min_value - mean) / stddev
        beta = (max_value - mean) / stddev
        Z = norm.cdf(beta) - norm.cdf(alpha)

        expected_mean = mean + (norm.pdf(alpha) - norm.pdf(beta)) / Z * stddev
        expected_skewness = 0  # by construction

        expected_variance = stddev**2 * (
            1
            - ((beta * norm.pdf(beta) - alpha * norm.pdf(alpha)) / Z)
            - (norm.pdf(alpha) - norm.pdf(beta)) / Z
        )

        self.assertAlmostEqual(expected_mean, np.mean(actual_values), places=2)
        self.assertAlmostEqual(
            expected_variance, np.var(actual_values), places=1
        )
        self.assertAlmostEqual(
            expected_skewness, scipy.stats.skew(actual_values), places=1
        )

    def test_compare_in_distribution_standard_scale(self):
        sample_cnt = 10000
        actual_values = keras.ops.convert_to_numpy(
            truncated_normal([sample_cnt], seed=1)
        )
        mean = 0
        stddev = 1.0
        # make sure that we are not returning values
        # outside of +- 2 * sigma range
        expected_min_value = -2 * stddev + mean
        expected_max_value = 2 * stddev + mean

        self.assertLessEqual(
            expected_min_value,
            actual_values.min(),
            msg="Outside of two standard deviations",
        )
        self.assertGreaterEqual(
            expected_max_value,
            actual_values.max(),
            msg="Outside of two standard deviations",
        )

        self.compare_three_moments_of_distributions(
            actual_values,
            mean=mean,
            stddev=stddev,
            min_value=expected_min_value,
            max_value=expected_max_value,
        )

    def test_compare_in_distribution_custom_scale(self):
        sample_cnt = 10000

        mean = 5.0
        stddev = 3
        # ensure that we are not returning values outside `+-2 * sigma` range
        expected_min_value = -2 * stddev + mean
        expected_max_value = 2 * stddev + mean

        actual_values = truncated_normal(
            [sample_cnt], seed=1, mean=mean, stddev=stddev
        ).numpy()
        self.assertLessEqual(
            expected_min_value,
            actual_values.min(),
            msg="Outside of two standard deviations",
        )
        self.assertGreaterEqual(
            expected_max_value,
            actual_values.max(),
            msg="Outside of two standard deviations",
        )

        self.compare_three_moments_of_distributions(
            actual_values,
            mean=mean,
            stddev=stddev,
            min_value=expected_min_value,
            max_value=expected_max_value,
        )

    def test_compare_in_distributions_with_actual(self):
        # make sure that we get min and max values correctly.
        sample_cnt = 20000

        mean = 5.0
        stddev = 3

        expected_values = keras.random.truncated_normal(
            [sample_cnt], seed=1, mean=mean, stddev=stddev
        ).numpy()
        actual_values = truncated_normal(
            [sample_cnt], seed=1, mean=mean, stddev=stddev
        ).numpy()
        self.assertAlmostEqual(
            expected_values.min(),
            actual_values.min(),
            places=2,
            msg="Boundaries are different from TF/Keras implementation",
        )
        self.assertAlmostEqual(
            expected_values.max(),
            actual_values.max(),
            places=2,
            msg="Boundaries are different from TF/Keras implementation",
        )

    def test_no_seed(self):
        with self.assertRaises(ValueError):
            truncated_normal([1])


if __name__ == "__main__":
    unittest.main()
