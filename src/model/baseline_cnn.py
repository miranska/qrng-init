import logging
import keras
from trainer import get_initializer

log = logging.getLogger(__name__)


def baseline_cnn(input_shape, output_shape, output_activation, model_config):
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
    x = keras.layers.Conv2D(
        model_config["units"],
        (3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer(),
    )(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        2 * model_config["units"],
        (3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer(),
    )(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        2 * model_config["units"],
        (3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer(),
    )(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(
        output_shape,
        activation=output_activation,
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer(),
        name="predictions",
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs)
