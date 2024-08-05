import pytest

from train_and_eval import get_starting_dim_id


def test_increment_start():
    assert get_starting_dim_id(1, "baseline-ann", "increment-start") == 1
    assert get_starting_dim_id(2, "baseline-ann", "increment-start") == 2
    assert (
        get_starting_dim_id(1, "baseline-cnn", "increment-start", min_dim_id=2)
        == 2
    )
    assert (
        get_starting_dim_id(
            3, "baseline-lstm", "increment-start", min_dim_id=2
        )
        == 4
    )


def test_increment_end_baseline_ann():
    assert get_starting_dim_id(1, "baseline-ann", "increment-end") == 1
    assert get_starting_dim_id(2, "baseline-ann", "increment-end") == 4
    assert get_starting_dim_id(3, "baseline-ann", "increment-end") == 7


def test_increment_end_baseline_cnn():
    assert get_starting_dim_id(1, "baseline-cnn", "increment-end") == 1
    assert get_starting_dim_id(2, "baseline-cnn", "increment-end") == 5
    assert get_starting_dim_id(3, "baseline-cnn", "increment-end") == 9


def test_increment_end_baseline_lstm():
    assert get_starting_dim_id(1, "baseline-lstm", "increment-end") == 1
    assert get_starting_dim_id(2, "baseline-lstm", "increment-end") == 3
    assert get_starting_dim_id(3, "baseline-lstm", "increment-end") == 5


def test_increment_end_baseline_transformer():
    assert get_starting_dim_id(1, "baseline-transformer", "increment-end") == 1
    assert get_starting_dim_id(2, "baseline-transformer", "increment-end") == 8
    assert (
        get_starting_dim_id(3, "baseline-transformer", "increment-end") == 15
    )


def test_invalid_sequence_scheme():
    with pytest.raises(ValueError, match="Unknown sequence scheme"):
        get_starting_dim_id(1, "baseline-ann", "unknown-scheme")


def test_invalid_model_id():
    with pytest.raises(ValueError, match="Unknown model unknown-model"):
        get_starting_dim_id(1, "unknown-model", "increment-end")
