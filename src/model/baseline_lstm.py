import logging
import keras
from trainer import get_initializer

log = logging.getLogger(__name__)


def baseline_lstm(
    input_shape,
    output_shape,
    output_activation,
    model_config,
):
    if len(input_shape) == 1:
        input_shape = (input_shape[0],)

    kernel_initializer = get_initializer(
        model_config["kernel_initializer"],
        initializer_type=model_config["initializer_type"],
    )

    recurrent_initializer = get_initializer(
        model_config["recurrent_initializer"],
        initializer_type=model_config["initializer_type"],
    )

    bias_initializer = get_initializer(
        model_config["bias_initializer"],
        initializer_type=model_config["initializer_type"],
    )

    embedding_initializer = get_initializer(
        "random-uniform",
        initializer_type="pseudo-random",  # use PRNG for both
    )

    inputs = keras.Input(shape=input_shape, name="input_layer")
    # Embed each token in a model_config["embedding_dim"]-dimensional vector
    x = keras.layers.Embedding(
        input_dim=model_config["max_features"],
        output_dim=model_config["embedding_dim"],
        embeddings_initializer=embedding_initializer(),
        name="embedding",
    )(inputs)
    x = keras.layers.LSTM(
        model_config["units"],
        return_sequences=False,
        kernel_initializer=kernel_initializer(),
        recurrent_initializer=recurrent_initializer(),
        bias_initializer=bias_initializer(),
        dropout=0.0,
        name="lstm",
    )(x)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(
        output_shape,
        activation=output_activation,
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer(),
        name="predictions",
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
