import keras
import pytest
from trainer import get_initializer
from model.baseline_transformer import (
    baseline_transformer,
)  # Adjust this import to match your actual module structure

# Lists of test cases for each parameter
initializers = [
    "glorot-normal",
    "glorot-uniform",
    "he-normal",
    "he-uniform",
    "lecun-normal",
    "lecun-uniform",
    "orthogonal",
    "random-normal",
    "random-uniform",
    "truncated-normal",
]
initializer_types = ["pseudo-random", "quasi-random"]

units = 8
num_heads = 2


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_transformer_shape(initializer, initializer_type):
    input_shape = (100,)  # Typically a sequence length for Transformer
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
        "max_features": 20000,  # Vocabulary size
        "embedding_dim": 128,  # Embedding dimensions
    }

    model = baseline_transformer(
        input_shape, output_shape, output_activation, model_config, num_heads
    )

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_transformer_compile(initializer, initializer_type):
    input_shape = (100,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
        "max_features": 20000,
        "embedding_dim": 128,
    }

    model = baseline_transformer(
        input_shape, output_shape, output_activation, model_config, num_heads
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")

    assert model.optimizer is not None
    assert model.loss == "binary_crossentropy"


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_transformer_initializers(initializer, initializer_type):
    input_shape = (100,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
        "max_features": 20000,
        "embedding_dim": 128,
    }

    model = baseline_transformer(
        input_shape, output_shape, output_activation, model_config, num_heads
    )

    kernel_initializer = get_initializer(
        model_config["kernel_initializer"],
        initializer_type=model_config["initializer_type"],
    )()

    bias_initializer = get_initializer(
        model_config["bias_initializer"],
        initializer_type=model_config["initializer_type"],
    )()

    embedding_layer = model.get_layer(
        index=1
    )  # TokenAndPositionEmbedding layer
    transformer_layer = model.get_layer(index=2)  # TransformerEncoder layer
    dense_layer = model.get_layer(name="predictions")

    assert isinstance(
        embedding_layer.embeddings_initializer,
        keras.src.initializers.RandomUniform,
    )
    assert isinstance(
        transformer_layer.kernel_initializer, type(kernel_initializer)
    )
    assert isinstance(
        transformer_layer.bias_initializer, type(bias_initializer)
    )
    assert isinstance(dense_layer.kernel_initializer, type(kernel_initializer))
    assert isinstance(dense_layer.bias_initializer, type(bias_initializer))


def test_baseline_transformer_structure():
    input_shape = (100,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": "glorot-normal",
        "bias_initializer": "zeros",
        "initializer_type": "pseudo-random",
        "max_features": 20000,
        "embedding_dim": 128,
    }

    model = baseline_transformer(
        input_shape, output_shape, output_activation, model_config, num_heads
    )

    print(model.summary())

    layers_m = model.layers
    assert len(layers_m) == 5
    assert layers_m[0].name == "input_layer"
    assert layers_m[1].name == "token_and_position_embedding"
    assert layers_m[2].name.startswith("transformer_encoder")
    assert layers_m[3].name == "flatten"
    assert layers_m[4].name == "predictions"
