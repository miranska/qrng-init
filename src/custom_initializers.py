import math
import logging

import keras
from keras.src import ops
from keras.src.initializers.random_initializers import compute_fans

from distributions_qr import random_normal, random_uniform, truncated_normal
from global_config import get_suggested_seed_value

log = logging.getLogger(__name__)


def _validate_initializer_args(
    initializer_name: str, scale: float, mode: str, distribution: str
) -> bool:
    """
    Validate the arguments of the variance scaling initializers using a mapping
    between the initializer name and the expected arguments.

    :param initializer_name: name of the initializer
    :param scale: scaling factor
    :param mode: mode of the initializer
    :param distribution: distribution of the initializer
    :return: `True` if the arguments are valid, `False` otherwise
    """
    initializer_args = {
        "GlorotUniformQR": {
            "scale": 1.0,
            "mode": "fan_avg",
            "distribution": "uniform",
        },
        "GlorotNormalQR": {
            "scale": 1.0,
            "mode": "fan_avg",
            "distribution": "truncated_normal",
        },
        "LecunNormalQR": {
            "scale": 1.0,
            "mode": "fan_in",
            "distribution": "truncated_normal",
        },
        "LecunUniformQR": {
            "scale": 1.0,
            "mode": "fan_in",
            "distribution": "uniform",
        },
        "HeNormalQR": {
            "scale": 2.0,
            "mode": "fan_in",
            "distribution": "truncated_normal",
        },
        "HeUniformQR": {
            "scale": 2.0,
            "mode": "fan_in",
            "distribution": "uniform",
        },
    }
    if initializer_name in initializer_args:
        valid_args = initializer_args[initializer_name]
        for arg_name, arg_value in zip(
            ["scale", "mode", "distribution"], [scale, mode, distribution]
        ):
            if arg_value != valid_args[arg_name]:
                log.error(
                    f"Argument {arg_name} for initializer {initializer_name} "
                    "is expected to be {valid_args[arg_name]}, "
                    f"but {arg_value} is provided."
                )
                return False
    else:
        log.error(
            f"Initializer {initializer_name} is not in the list of "
            "variance scaling initializers."
        )
        return False
    return True


def _get_seed_value(initializer_name: str) -> int:
    """
    Get seed value for a given initializer.
    For a given initializer_name, seed value start from zero
    and increments sequentially for every new call.

    :param initializer_name: name of the initializer
    :return: seed value
    """
    seed = get_suggested_seed_value(initializer_name)
    log.info(f"Setting seed value for {initializer_name} to {seed}.")
    return seed


# The code below is based on the keras.src.initializers.random_initializers
# module for Keras 3.3.3


class RandomNormalQR(keras.initializers.RandomNormal):
    """Random normal initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    Draws samples from a normal distribution for given parameters.

    Examples:

    >>> # Standalone usage:
    >>> initializer = RandomNormalQR(mean=0.0, stddev=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomNormalQR(mean=0.0, stddev=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented
    """

    def __call__(self, shape, dtype=None):
        return random_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=_get_seed_value(initializer_name=self.__class__.__name__),
            dtype=dtype,
        )


class TruncatedNormalQR(keras.initializers.TruncatedNormal):
    """Initializer that generates a truncated normal distribution.

    Inherits from the Keras class, but draws values from Sobol sequences.

    The values generated are similar to values from a
    `RandomNormal` initializer, except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.

    Examples:

    >>> # Standalone usage:
    >>> initializer = TruncatedNormalQR(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = TruncatedNormalQR(mean=0., stddev=1.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented
    """

    def __call__(self, shape, dtype=None):
        return truncated_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=_get_seed_value(initializer_name=self.__class__.__name__),
            dtype=dtype,
        )


class RandomUniformQR(keras.initializers.RandomUniform):
    """Random uniform initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    Draws samples from a uniform distribution for given parameters.

    Examples:

    >>> # Standalone usage:
    >>> initializer = RandomUniformQR(minval=0.0, maxval=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomUniformQR(minval=0.0, maxval=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        minval: A python scalar or a scalar keras tensor. Lower bound of the
            range of random values to generate (inclusive).
        maxval: A python scalar or a scalar keras tensor. Upper bound of the
            range of random values to generate (exclusive).
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented

    """

    def __call__(self, shape, dtype=None):
        return random_uniform(
            shape=shape,
            minval=self.minval,
            maxval=self.maxval,
            seed=_get_seed_value(initializer_name=self.__class__.__name__),
            dtype=dtype,
        )


class VarianceScalingQR(keras.initializers.VarianceScaling):
    """Initializer that adapts its scale to the shape of its input tensors.

    Inherits from the Keras class, but draws values from Sobol sequences.

    With `distribution="truncated_normal" or "untruncated_normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero
    and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
    n)`, where `n` is:

    - number of input units in the weight tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`

    With `distribution="uniform"`, samples are drawn from a uniform
    distribution
    within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = VarianceScalingQR(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = VarianceScalingQR(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        scale: Scaling factor (positive float).
        mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
        distribution: Random distribution to use.
            One of `"truncated_normal"`, `"untruncated_normal"`, or
            `"uniform"`.
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented
    """

    def __call__(self, shape, dtype=None):
        scale = self.scale
        fan_in, fan_out = compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == "truncated_normal":
            stddev = math.sqrt(scale) / 0.87962566103423978
            return truncated_normal(
                shape,
                mean=0.0,
                stddev=stddev,
                dtype=dtype,
                seed=_get_seed_value(initializer_name=self.__class__.__name__),
            )
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return random_normal(
                shape,
                mean=0.0,
                stddev=stddev,
                dtype=dtype,
                seed=_get_seed_value(initializer_name=self.__class__.__name__),
            )
        else:
            limit = math.sqrt(3.0 * scale)
            return random_uniform(
                shape,
                minval=-limit,
                maxval=limit,
                dtype=dtype,
                seed=_get_seed_value(initializer_name=self.__class__.__name__),
            )


class GlorotUniformQR(VarianceScalingQR):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
    units in the weight tensor and `fan_out` is the number of output units).

    Examples:

    >>> # Standalone usage:
    >>> initializer = GlorotUniformQR()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotUniformQR()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(
        self, seed=None, scale=1.0, mode="fan_avg", distribution="uniform"
    ):
        assert _validate_initializer_args(
            initializer_name=self.__class__.__name__,
            scale=scale,
            mode=mode,
            distribution=distribution,
        )
        super().__init__(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )


class GlorotNormalQR(VarianceScalingQR):
    """The Glorot normal initializer, also called Xavier normal initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
    input units in the weight tensor and `fan_out` is the number of output
    units in the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = GlorotNormalQR()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotNormalQR()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(
        self,
        seed=None,
        scale=1.0,
        mode="fan_avg",
        distribution="truncated_normal",
    ):
        assert _validate_initializer_args(
            initializer_name=self.__class__.__name__,
            scale=scale,
            mode=mode,
            distribution=distribution,
        )
        super().__init__(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )


class LecunNormalQR(VarianceScalingQR):
    """Lecun normal initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    Initializers allow you to pre-specify an initialization strategy, encoded
    in the Initializer object, without knowing the shape and dtype of the
    variable being initialized.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = LecunNormalQR()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunNormalQR()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(
        self,
        seed=None,
        scale=1.0,
        mode="fan_in",
        distribution="truncated_normal",
    ):
        assert _validate_initializer_args(
            initializer_name=self.__class__.__name__,
            scale=scale,
            mode=mode,
            distribution=distribution,
        )
        super().__init__(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )


class LecunUniformQR(VarianceScalingQR):
    """Lecun uniform initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:

    >>> # Standalone usage:
    >>> initializer = LecunUniformQR()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunUniformQR()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented


    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(
        self, seed=None, scale=1.0, mode="fan_in", distribution="uniform"
    ):
        assert _validate_initializer_args(
            initializer_name=self.__class__.__name__,
            scale=scale,
            mode=mode,
            distribution=distribution,
        )
        super().__init__(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )


class HeNormalQR(VarianceScalingQR):
    """He normal initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = HeNormalQR()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeNormalQR()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented


    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(
        self,
        seed=None,
        scale=2.0,
        mode="fan_in",
        distribution="truncated_normal",
    ):
        assert _validate_initializer_args(
            initializer_name=self.__class__.__name__,
            scale=scale,
            mode=mode,
            distribution=distribution,
        )
        super().__init__(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )


class HeUniformQR(VarianceScalingQR):
    """He uniform variance scaling initializer.

    Inherits from the Keras class, but draws values from Sobol sequences.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:

    >>> # Standalone usage:
    >>> initializer = HeUniformQR()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeUniformQR()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented


    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(
        self,
        seed=None,
        scale=2.0,
        mode="fan_in",
        distribution="uniform",
    ):
        assert _validate_initializer_args(
            initializer_name=self.__class__.__name__,
            scale=scale,
            mode=mode,
            distribution=distribution,
        )
        super().__init__(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )


class OrthogonalInitializerQR(keras.initializers.OrthogonalInitializer):
    """Initializer that generates an orthogonal matrix.

    Inherits from the Keras class, but draws values from Sobol sequences.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Examples:

    >>> # Standalone usage:
    >>> initializer = keras.initializers.OrthogonalQR()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = keras.initializers.OrthogonalQR()
    >>> layer = keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: The code ignores the seed value -- each time the initializer
           is called, the seed will be incremented


    Reference:

    - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
    """

    def __call__(self, shape, dtype=None):
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be "
                "at least two-dimensional. Received: "
                f"shape={shape} of rank {len(shape)}."
            )

        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

        # Generate a random matrix
        a = random_normal(
            flat_shape,
            seed=_get_seed_value(initializer_name=self.__class__.__name__),
            dtype=dtype,
        )
        # Compute the qr factorization
        q, r = ops.qr(a)
        # Make Q uniform
        d = ops.diag(r)
        q *= ops.sign(d)
        if num_rows < num_cols:
            q = ops.transpose(q)
        return self.gain * ops.reshape(q, shape)
