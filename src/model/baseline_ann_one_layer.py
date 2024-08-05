import logging

import keras.src.initializers
import keras
from trainer import get_initializer

log = logging.getLogger(__name__)


def baseline_ann_one_layer(
    input_shape, output_shape, output_activation, model_config
):
    if len(input_shape) == 1:
        input_shape = (input_shape[0],)

    kernel_initializer = get_initializer(
        model_config["kernel_initializer"],
        initializer_type=model_config["initializer_type"],
    )

    bias_initializer = get_initializer(
        model_config["bias_initializer"],
        initializer_type=model_config["initializer_type"],
    )

    inputs = keras.Input(shape=input_shape, name="input_layer")
    x = keras.layers.Dense(
        model_config["units"],
        activation="relu",
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer(),
        name="dense_1",
    )(inputs)
    outputs = keras.layers.Dense(
        output_shape,
        activation=output_activation,
        kernel_initializer=keras.initializers.GlorotUniform,
        bias_initializer=keras.initializers.Zeros,
        name="predictions",
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs)
