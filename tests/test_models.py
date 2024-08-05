import pytest

from models import ModelBuilder

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

models_with_flat_input = [
    "baseline-ann",
    "baseline-ann-one-layer",
    "baseline-lstm",
    "baseline-transformer",
]

models_with_3d_input = [
    "baseline-cnn",
]

initializer_types = ["pseudo-random", "quasi-random"]


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("model", models_with_flat_input)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_ability_to_create_model_for_flattened_data(
    initializer, model, initializer_type
):
    try:
        model_config = {
            "initializer_type": initializer_type,
            "kernel_initializer": initializer,
            "bias_initializer": "zeros",
            "recurrent_initializer": "orthogonal",  # used only by LSTM
            "max_features": 10,  # for IMDB dataset
            "units": 8,  # Used by ANNs, LSTM, and Transformer
            "embedding_dim": 32,  # Used by LSTM and Transformers
        }
        output_shape = 10
        input_shape = [512]
        model_builder = ModelBuilder(
            model_id=model,
            input_shape=input_shape,
            output_shape=output_shape,
            output_activation="softmax",
            model_config=model_config,
        )
        model = model_builder.get_model()

    except Exception as e:
        pytest.fail(
            f"Model builder crashed with inputs "
            f"{initializer}, {model}, {initializer_type}: {str(e)}"
        )


@pytest.mark.parametrize("initializer", initializers)
@pytest.mark.parametrize("model", models_with_3d_input)
@pytest.mark.parametrize("initializer_type", initializer_types)
def test_ability_to_create_model_for_3d_data(
    initializer, model, initializer_type
):
    try:
        model_config = {
            "initializer_type": initializer_type,
            "kernel_initializer": initializer,
            "bias_initializer": "zeros",
            "recurrent_initializer": "orthogonal",  # used only by LSTM
            "max_features": 10,  # for IMDB dataset
            "units": 8,  # Used by LSTM and 2-layered ANN
        }
        output_shape = 10
        input_shape = [32, 32, 3]
        model_builder = ModelBuilder(
            model_id=model,
            input_shape=input_shape,
            output_shape=output_shape,
            output_activation="softmax",
            model_config=model_config,
        )
        model = model_builder.get_model()

    except Exception as e:
        pytest.fail(
            f"Model builder crashed with inputs "
            f"{initializer}, {model}, {initializer_type}: {str(e)}"
        )


def test_unknown_model_handling():
    with pytest.raises(ValueError) as exe_info:
        model_builder = ModelBuilder(
            model_id="blah",
            input_shape=[1],
            output_shape=2,
            output_activation="softmax",
            model_config={},
        )
        model_builder.get_model()
    assert str(exe_info.value) == "Unknown model_id blah"
