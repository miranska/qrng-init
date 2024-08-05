import pytest
from trainer import get_initializer
from model.baseline_cnn import (
    baseline_cnn,
)

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
def test_baseline_cnn_shape(initializer, initializer_type):
    input_shape = (28, 28, 1)
    output_shape = 10
    output_activation = "softmax"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
    }

    model = baseline_cnn(
        input_shape, output_shape, output_activation, model_config
    )

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_cnn_compile(initializer, initializer_type):
    input_shape = (28, 28, 1)
    output_shape = 10
    output_activation = "softmax"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
    }

    model = baseline_cnn(
        input_shape, output_shape, output_activation, model_config
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_cnn_initializers(initializer, initializer_type):
    input_shape = (28, 28, 1)
    output_shape = 10
    output_activation = "softmax"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
    }

    model = baseline_cnn(
        input_shape, output_shape, output_activation, model_config
    )

    kernel_initializer = get_initializer(
        model_config["kernel_initializer"],
        initializer_type=model_config["initializer_type"],
    )()

    bias_initializer = get_initializer(
        model_config["bias_initializer"],
        initializer_type=model_config["initializer_type"],
    )()

    conv_layer_1 = model.get_layer(index=1)
    conv_layer_2 = model.get_layer(index=3)
    conv_layer_3 = model.get_layer(index=5)
    dense_layer = model.get_layer(name="predictions")

    assert isinstance(
        conv_layer_1.kernel_initializer, type(kernel_initializer)
    )
    assert isinstance(conv_layer_1.bias_initializer, type(bias_initializer))
    assert isinstance(
        conv_layer_2.kernel_initializer, type(kernel_initializer)
    )
    assert isinstance(conv_layer_2.bias_initializer, type(bias_initializer))
    assert isinstance(
        conv_layer_3.kernel_initializer, type(kernel_initializer)
    )
    assert isinstance(conv_layer_3.bias_initializer, type(bias_initializer))
    assert isinstance(dense_layer.kernel_initializer, type(kernel_initializer))
    assert isinstance(dense_layer.bias_initializer, type(bias_initializer))


def test_baseline_cnn_structure():
    input_shape = (28, 28, 1)
    output_shape = 10
    output_activation = "softmax"
    model_config = {
        "units": units,
        "kernel_initializer": "glorot-normal",
        "bias_initializer": "zeros",
        "initializer_type": "pseudo-random",
    }

    model = baseline_cnn(
        input_shape, output_shape, output_activation, model_config
    )

    layers = model.layers
    assert (
        len(layers) == 8
    )  # input_layer, 3 Conv2D, 2 MaxPooling2D, Flatten, predictions
    assert layers[0].name == "input_layer"
    assert layers[1].name.startswith("conv2d")
    assert layers[2].name.startswith("max_pooling2d")
    assert layers[3].name.startswith("conv2d")
    assert layers[4].name.startswith("max_pooling2d")
    assert layers[5].name.startswith("conv2d")
    assert layers[6].name.startswith("flatten")
    assert layers[7].name == "predictions"
