import keras
from keras import layers
import keras_nlp
import logging
from trainer import get_initializer

log = logging.getLogger(__name__)

"""
Reference: https://github.com/keras-team/keras-io/blob/b8bb4c0/
Path: examples/nlp/text_classification_with_transformer.py

The following implementation is a simplified version
of the given Keras example combined with keras_nlp tutorial from
https://keras.io/guides/keras_nlp/transformer_pretraining/
"""


def baseline_transformer(
    input_shape,
    output_shape,
    output_activation,
    model_config,
    num_heads=2,  # Number of attention heads
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

    embedding_initializer = get_initializer(
        "random-uniform",
        initializer_type="pseudo-random",  # use PRNG for both
    )

    max_len = input_shape[0]
    inputs = layers.Input(shape=(max_len,), name="input_layer")
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        sequence_length=max_len,
        vocabulary_size=model_config["max_features"],
        embedding_dim=model_config["embedding_dim"],
        embeddings_initializer=embedding_initializer(),
    )(inputs)
    transformer_encoder = keras_nlp.layers.TransformerEncoder(
        num_heads=num_heads,
        # Hidden layer size in feed forward network inside transformer
        intermediate_dim=model_config["units"],
        dropout=0.1,
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer,
    )
    x = transformer_encoder(embedding_layer)
    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(
        output_shape,
        activation=output_activation,
        kernel_initializer=kernel_initializer(),
        bias_initializer=bias_initializer(),
        name="predictions",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
