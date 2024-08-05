from unittest.mock import patch, MagicMock

from train_and_eval import get_starting_dim_id_auto


@patch("random.sample")
@patch("train_and_eval.build_train_and_evaluate_model")
def test_get_starting_dim_id_auto(
    mock_build_train_and_evaluate_model, mock_random_sample
):
    # Setup
    mock_random_sample.return_value = [1, 2, 3, 4, 5]
    mock_build_train_and_evaluate_model.side_effect = [
        {"summary_metrics": {"test-categorical_accuracy": [0.8]}},
        {"summary_metrics": {"test-categorical_accuracy": [0.85]}},
        {"summary_metrics": {"test-categorical_accuracy": [0.75]}},
        {"summary_metrics": {"test-categorical_accuracy": [0.9]}},
        {"summary_metrics": {"test-categorical_accuracy": [0.65]}},
        {"summary_metrics": {"test-categorical_accuracy": [0.81]}},
        {"summary_metrics": {"test-categorical_accuracy": [0.86]}},
        {
            "summary_metrics": {"test-categorical_accuracy": [0.95]}
        },  # best value mapped to seed 4, try 2
        {"summary_metrics": {"test-categorical_accuracy": [0.25]}},
        {"summary_metrics": {"test-categorical_accuracy": [0.64]}},
    ]

    # Parameters
    auto_epoch_count = 1
    auto_max_seed_value = 10
    auto_min_seed_value = 1
    auto_seeds_count = 5
    auto_repetition_count = 2
    build_cfg = {
        "batch_size": 32,
        "input_shape": (28, 28, 1),
        "learning_rate": 0.001,
        "model_config": {},
        "model_id": "baseline-ann",
        "optimizer": "adam",
        "output_shape": (10,),
        "test_dg": MagicMock(),
        "train_dg": MagicMock(),
        "val_dg": MagicMock(),
    }

    # Call function
    result = get_starting_dim_id_auto(
        auto_epoch_count,
        auto_max_seed_value,
        auto_min_seed_value,
        auto_seeds_count,
        auto_repetition_count,
        build_cfg,
    )

    # Assertions
    assert result == 4  # Best accuracy (0.9) with seed 4
    mock_random_sample.assert_called_once_with(
        range(auto_min_seed_value, auto_max_seed_value + 1), auto_seeds_count
    )


@patch("random.sample")
@patch("train_and_eval.build_train_and_evaluate_model")
def test_get_starting_dim_id_auto_single_seed(
    mock_build_train_and_evaluate_model, mock_random_sample
):
    # Setup
    mock_random_sample.return_value = [3]
    mock_build_train_and_evaluate_model.return_value = {
        "summary_metrics": {"test-categorical_accuracy": [0.8]}
    }

    # Parameters
    auto_epoch_count = 1
    auto_max_seed_value = 10
    auto_min_seed_value = 1
    auto_seeds_count = 1
    auto_repetition_count = 1
    build_cfg = {
        "batch_size": 32,
        "input_shape": (28, 28, 1),
        "learning_rate": 0.001,
        "model_config": {},
        "model_id": "baseline-ann",
        "optimizer": "adam",
        "output_shape": (10,),
        "test_dg": MagicMock(),
        "train_dg": MagicMock(),
        "val_dg": MagicMock(),
    }

    # Call function
    result = get_starting_dim_id_auto(
        auto_epoch_count,
        auto_max_seed_value,
        auto_min_seed_value,
        auto_seeds_count,
        auto_repetition_count,
        build_cfg,
    )

    # Assertions
    assert result == 3  # Only one seed to choose from
    mock_random_sample.assert_called_once_with(
        range(auto_min_seed_value, auto_max_seed_value + 1), auto_seeds_count
    )
