import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from custom_initializers import RandomUniformQR
from distributions_qr import get_sobol_sequence_for_specific_dimension
from global_config import reset_suggested_seed_values


def test_sobol_sequence_0():
    expected = np.array([0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875])
    actual = get_sobol_sequence_for_specific_dimension(
        dim_id=1, num_results=8
    ).numpy()
    np.testing.assert_equal(actual, expected)


def test_sobol_sequence_1():
    expected = np.array([0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625, 0.3125])
    actual = get_sobol_sequence_for_specific_dimension(
        dim_id=2, num_results=8
    ).numpy()
    np.testing.assert_equal(actual, expected)


def test_dense_layer_init_random_uniform():
    reset_suggested_seed_values()
    model = Sequential()
    model.add(
        Dense(
            units=4,
            activation="relu",
            input_shape=(1,),
            kernel_initializer=RandomUniformQR(minval=0.0, maxval=1.0),
            bias_initializer="zero",
        )
    )
    # Initialize the model by building it
    model.build()

    # Access the weights of the dense layer
    weights = model.layers[0].get_weights()

    kernel_weights = weights[0]
    bias_weights = weights[1]

    np.testing.assert_equal(
        kernel_weights,
        # np.array([[0.5, 0.25, 0.75, 0.375]]), # for dim 1
        np.array([[0.5, 0.75, 0.25, 0.375]]),  # for dim 0
    )

    np.testing.assert_equal(
        bias_weights,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )


def test_dense_layer_init_random_uniform_two_input():
    reset_suggested_seed_values()
    model = Sequential()
    model.add(
        Dense(
            units=4,
            activation="relu",
            input_shape=(2,),
            kernel_initializer=RandomUniformQR(minval=0.0, maxval=1.0),
            bias_initializer="zero",
        )
    )
    # Initialize the model by building it
    model.build()

    # Access the weights of the dense layer
    weights = model.layers[0].get_weights()

    kernel_weights = weights[0]
    bias_weights = weights[1]

    np.testing.assert_equal(
        kernel_weights,
        # np.array(
        #     [
        #         [0.5, 0.25, 0.75, 0.375],
        #         [0.875, 0.125, 0.625, 0.3125],
        #     ]
        # ), for dim 1
        np.array(
            [
                [0.5, 0.75, 0.25, 0.375],
                [0.875, 0.625, 0.125, 0.1875],
            ]
        ),
    )

    np.testing.assert_equal(
        bias_weights,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
