import pytest
from model.baseline_ann import baseline_ann
from trainer import get_initializer

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
def test_baseline_ann_shape(initializer, initializer_type):
    input_shape = (20,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
    }

    model = baseline_ann(
        input_shape, output_shape, output_activation, model_config
    )

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_ann_compile(initializer, initializer_type):
    input_shape = (20,)
    output_shape = 1
    output_activation = "sigmoid"
    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
    }

    model = baseline_ann(
        input_shape, output_shape, output_activation, model_config
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")

    assert model.optimizer is not None
    assert model.loss == "binary_crossentropy"


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_baseline_ann_initializers(initializer, initializer_type):
    input_shape = (20,)
    output_shape = 1
    output_activation = "sigmoid"

    model_config = {
        "units": units,
        "kernel_initializer": initializer,
        "bias_initializer": "zeros",
        "initializer_type": initializer_type,
    }

    model = baseline_ann(
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

    dense_layer_1 = model.get_layer(name="dense_1")
    assert isinstance(
        dense_layer_1.kernel_initializer, type(kernel_initializer)
    )
    assert isinstance(dense_layer_1.bias_initializer, type(bias_initializer))

    dense_layer_2 = model.get_layer(name="dense_2")
    assert isinstance(
        dense_layer_2.kernel_initializer, type(kernel_initializer)
    )
    assert isinstance(dense_layer_2.bias_initializer, type(bias_initializer))


def test_baseline_ann_structure():
    input_shape = (20,)
    output_shape = 1
    output_activation = "sigmoid"

    model_config = {
        "units": units,
        "kernel_initializer": "glorot-normal",
        "bias_initializer": "zeros",
        "initializer_type": "pseudo-random",
    }

    model = baseline_ann(
        input_shape, output_shape, output_activation, model_config
    )

    layers = model.layers
    assert len(layers) == 4  # input_layer, dense_1, dense_2, predictions
    assert layers[0].name == "input_layer"
    assert layers[1].name == "dense_1"
    assert layers[2].name == "dense_2"
    assert layers[3].name == "predictions"
