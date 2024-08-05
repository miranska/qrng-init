import warnings

import numpy as np
import keras.src.ops as ops
from keras.src.backend import floatx
from scipy.stats import norm, qmc

"""
Contains a collection of functions to draw from multiple distributions with
quasi-random sequences used as an underlying source of randomness.
"""


def get_sobol_sequence_for_specific_dimension(
    dim_id: int, num_results: int, skip: int = 0, dtype=None
):
    """
    Get Sobol sequences for a specific dimension. The code is no efficient.
    It performs `O(dim_id * num_results)` rather than `O(num_results)` due to
    the limitations of the Scipy API for the Sobol sequences.

    Note that Tensorflow skips the first zero value of the Sobol sequences.

    :param dim_id: Positive scalar representing id of the Sobol sequence
           dimension in the range [1, 21201].
    :param num_results: Positive scalar Tensor of dtype int32.
           The number of Sobol points to return in the output.
    :param skip: (Optional) Positive scalar Tensor of dtype int32.
           The number of initial points of the Sobol sequence to skip.
           Default value is 0.
    :param dtype: (Optional) The dtype of the sample.
           It is usually one of the floating types.
           Defaults to the output of `keras.src.backend.floatx()`
           if set to `None`.
    :return: 1-D tensor
    """
    dim_id = int(dim_id)
    num_results = ops.convert_to_numpy(num_results)
    skip = ops.convert_to_numpy(skip)

    dtype = dtype or floatx()

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=(
            "The balance properties of Sobol' points "
            "require n to be a power of 2.*"
        ),
    )
    # sample extra 1 value to drop 0th element and
    # sample skip elements that should be avoided
    sequences = qmc.Sobol(d=dim_id, scramble=False).random(
        n=(num_results + 1 + skip)
    )[:, dim_id - 1]

    # remove the first element to align it with
    # the TensorFlow Sobol sequences implementation
    sequences = sequences[1:]

    # add the option to skip items
    sequences = sequences[skip:]

    return ops.convert_to_tensor(sequences, dtype=dtype)


def random_uniform(shape, minval=0, maxval=1.0, dtype=None, seed: int = None):
    """
    Returns a tensor with uniform distribution of values drawn from Sobol
    sequences.
    This function mimics API of the tf.keras.backend.random_uniform function

    :param shape: A tuple of integers, the shape of tensor to create.
    :param minval: A float, lower boundary of the uniform distribution to draw
           samples.
    :param maxval: A float, upper boundary of the uniform distribution to draw
           samples.
    :param dtype: (Optional) The dtype of the sample.
           It is usually one of the floating types.
           Defaults to the output of `keras.src.backend.floatx()`
           if set to `None`.
    :param seed: Integer, random seed.
           It corresponds to the id of the Sobol sequence dimension in the
           range [0, infinity).
    :return: A tensor.
    """

    if seed is None:
        raise ValueError(
            "Please provide the seed value. "
            "`None` values are allowed only for API compatibility."
        )

    shape = ops.convert_to_numpy(shape)
    minval = ops.convert_to_numpy(minval)
    maxval = ops.convert_to_numpy(maxval)
    seed = ops.convert_to_numpy(seed)

    dtype = dtype or floatx()
    elements_count = np.prod(shape)
    original_values = ops.convert_to_numpy(
        get_sobol_sequence_for_specific_dimension(
            dim_id=seed, num_results=elements_count, dtype=dtype
        )
    )
    rescaled_values = original_values * (maxval - minval) + minval
    rescaled_values = ops.cast(rescaled_values, dtype=dtype)
    return ops.reshape(x=rescaled_values, newshape=shape)


def random_normal(
    shape,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype=None,
    seed: int = None,
):
    """
    Returns a tensor with normal distribution of values.
    This function mimics API of the tf.keras.backend.random_normal function.

    Currently, we get the values by applying quantile functions to individual
    elements of the Sobol sequence. This is inefficient.

    :param shape: A tuple of integers, the shape of tensor to create.
    :param mean: A float, the mean value of the normal distribution to draw
           samples. Defaults to 0.0.
    :param stddev: A float, the standard deviation of the normal distribution
           to draw samples. Defaults to 1.0.
    :param dtype: (Optional) The dtype of the sample.
           It is usually one of the floating types.
           Defaults to the output of `keras.src.backend.floatx()`
           if set to `None`.
    :param seed: Integer, random seed.
           It corresponds to the id of the Sobol sequence dimension
           in the range [0, infinity).
    :return: A tensor.
    """
    if seed is None:
        raise ValueError(
            "Please provide the seed value. "
            "`None` values are allowed only for API compatibility."
        )
    shape = ops.convert_to_numpy(shape)
    mean = ops.convert_to_numpy(mean)
    stddev = ops.convert_to_numpy(stddev)
    seed = ops.convert_to_numpy(seed)

    dtype = dtype or floatx()
    elements_count = np.prod(shape)

    uniform_random_values = ops.convert_to_numpy(
        get_sobol_sequence_for_specific_dimension(
            dim_id=seed, num_results=elements_count, dtype=dtype
        )
    )
    normal_random_values = norm.ppf(
        uniform_random_values, loc=mean, scale=stddev
    )
    normal_random_values = ops.cast(normal_random_values, dtype=dtype)
    return ops.reshape(x=normal_random_values, newshape=shape)


def truncated_normal(
    shape,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype=None,
    seed: int = None,
):
    """
    Returns a tensor with truncated random normal distribution of values.
    The generated values follow a truncated normal distribution with specified
    mean and standard deviation,
    except that values whose magnitude is more than two standard deviations
    from the mean are dropped and re-picked.
    This function mimics API of the tf.keras.backend.random_normal function.

    Currently, we get the values by applying transformations to individual
    elements of the Sobol sequence using the inverse transform.
    This is inefficient.

    :param shape: A tuple of integers, the shape of tensor to create.
    :param mean: A float, the mean value of the normal distribution to draw
           samples. Defaults to 0.0.
    :param stddev: A float, the standard deviation of the normal distribution
           to draw samples. Defaults to 1.0.
    :param dtype: (Optional) The dtype of the sample.
           It is usually one of the floating types.
           Defaults to the output of `keras.src.backend.floatx()`
           if set to `None`.
    :param seed: Integer, random seed.
           It corresponds to the id of the Sobol sequence dimension in
           the range [0, infinity).
    :return: A tensor.
    """  # noqa: E501

    if seed is None:
        raise ValueError(
            "Please provide the seed value. "
            "`None` values are allowed only for API compatibility."
        )

    shape = ops.convert_to_numpy(shape)
    mean = ops.convert_to_numpy(mean)
    stddev = ops.convert_to_numpy(stddev)
    seed = ops.convert_to_numpy(seed)

    dtype = dtype or floatx()
    elements_count = np.prod(shape)
    mu = mean
    sigma = stddev
    a = -2.0 * sigma + mu
    b = 2.0 * sigma + mu
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    uniform_random_values = ops.convert_to_numpy(
        get_sobol_sequence_for_specific_dimension(
            dim_id=seed, num_results=elements_count, dtype=dtype
        )
    )

    truncated_normal_random_values = (
        norm.ppf(
            norm.cdf(alpha)
            + uniform_random_values * (norm.cdf(beta) - norm.cdf(alpha))
        )
        * sigma
        + mu
    )
    truncated_normal_random_values = ops.cast(
        truncated_normal_random_values, dtype=dtype
    )
    return ops.reshape(x=truncated_normal_random_values, newshape=shape)
