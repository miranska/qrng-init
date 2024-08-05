import pytest
import keras
from trainer import get_initializer
from model.baseline_lstm import (
    baseline_lstm,
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


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_lstm_shape(initializer, initializer_type):
    input_shape = (100,)  # Typically a sequence length for LSTM
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "recurrent_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
        "max_features": 20000,  # Vocabulary size
        "embedding_dim": 128,  # Embedding dimensions
    }

    model = baseline_lstm(
        input_shape, output_shape, output_activation, model_config
    )

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_lstm_compile(initializer, initializer_type):
    input_shape = (100,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "recurrent_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
        "max_features": 20000,
        "embedding_dim": 128,
    }

    model = baseline_lstm(
        input_shape, output_shape, output_activation, model_config
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")

    assert model.optimizer is not None
    assert model.loss == "binary_crossentropy"


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_lstm_initializers(initializer, initializer_type):
    input_shape = (100,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "recurrent_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
        "max_features": 20000,
        "embedding_dim": 128,
    }

    model = baseline_lstm(
        input_shape, output_shape, output_activation, model_config
    )

    kernel_initializer = get_initializer(
        model_config["kernel_initializer"],
        initializer_type=model_config["initializer_type"],
    )()

    recurrent_initializer = get_initializer(
        model_config["recurrent_initializer"],
        initializer_type=model_config["initializer_type"],
    )()

    bias_initializer = get_initializer(
        model_config["bias_initializer"],
        initializer_type=model_config["initializer_type"],
    )()

    embedding_layer = model.get_layer(name="embedding")
    lstm_layer = model.get_layer(name="lstm")
    dense_layer = model.get_layer(name="predictions")

    assert isinstance(
        embedding_layer.embeddings_initializer,
        keras.src.initializers.RandomUniform,
    )
    assert isinstance(lstm_layer.kernel_initializer, type(kernel_initializer))
    assert isinstance(
        lstm_layer.recurrent_initializer, type(recurrent_initializer)
    )
    assert isinstance(lstm_layer.bias_initializer, type(bias_initializer))
    assert isinstance(dense_layer.kernel_initializer, type(kernel_initializer))
    assert isinstance(dense_layer.bias_initializer, type(bias_initializer))


def test_baseline_lstm_structure():
    input_shape = (100,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": "glorot-normal",
        "recurrent_initializer": "orthogonal",
        "bias_initializer": "zeros",
        "initializer_type": "pseudo-random",
        "max_features": 20000,
        "embedding_dim": 128,
    }

    model = baseline_lstm(
        input_shape, output_shape, output_activation, model_config
    )

    layers = model.layers
    assert (
        len(layers) == 5
    )  # input_layer, embedding, lstm, flatten, predictions
    assert layers[0].name == "input_layer"
    assert layers[1].name == "embedding"
    assert layers[2].name == "lstm"
    assert layers[3].name == "flatten"
    assert layers[4].name == "predictions"
