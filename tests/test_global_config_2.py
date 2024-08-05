import pytest
import logging
from global_config import (
    reset_suggested_seed_values,
    get_suggested_seed_value,
    set_suggested_seed_value,
)


def test_default_reset():
    reset_suggested_seed_values()
    assert (
        get_suggested_seed_value("test") == 1
    ), "Default seed value should be 1 after reset"


def test_custom_reset():
    reset_suggested_seed_values(5)
    assert (
        get_suggested_seed_value("test") == 5
    ), "Seed value should start at the custom value 5"


def test_non_existent_initializer():
    reset_suggested_seed_values()
    assert (
        get_suggested_seed_value("new_initializer") == 1
    ), "Should start at default seed 1"


def test_non_existent_initializer_custom():
    reset_suggested_seed_values(5)
    assert (
        get_suggested_seed_value("new_initializer") == 5
    ), "Should start at custom seed 5"


def test_increment_on_access():
    reset_suggested_seed_values()
    initializer_name = "increment_test"
    first_value = get_suggested_seed_value(initializer_name)
    second_value = get_suggested_seed_value(initializer_name)
    assert (
        second_value == first_value + 1
    ), "Seed value should increment on each access"


def test_setting_new_value():
    reset_suggested_seed_values()
    initializer_name = "new_initializer"
    set_suggested_seed_value(initializer_name, 10)
    assert (
        get_suggested_seed_value(initializer_name) == 11
    ), "Seed value should be set to 11 as we do a forced increment"
    assert (
        get_suggested_seed_value(initializer_name) == 12
    ), "Seed value should be set to 12 as we do a forced increment"


def test_setting_new_value_with_override_no_forced_init():
    reset_suggested_seed_values()
    initializer_name = "new_initializer"

    assert (
        get_suggested_seed_value(initializer_name) == 1
    ), "Seed value should be set to 1 as we do a forced increment"

    set_suggested_seed_value(initializer_name, 15)
    assert (
        get_suggested_seed_value(initializer_name) == 2
    ), "Seed value should be set to 2 as we do a forced increment"

    set_suggested_seed_value(initializer_name, 5)
    assert (
        get_suggested_seed_value(initializer_name) == 3
    ), "Seed value should be set to 3 as we do a forced increment"


def test_setting_new_value_with_override():
    reset_suggested_seed_values()
    initializer_name = "new_initializer"
    set_suggested_seed_value(initializer_name, 10)

    assert (
        get_suggested_seed_value(initializer_name) == 11
    ), "Seed value should be set to 11 as we do a forced increment"

    set_suggested_seed_value(initializer_name, 15)
    assert (
        get_suggested_seed_value(initializer_name) == 12
    ), "Seed value should be set to 12 as we do a forced increment"

    set_suggested_seed_value(initializer_name, 5)
    assert (
        get_suggested_seed_value(initializer_name) == 13
    ), "Seed value should be set to 13 as we do a forced increment"


def test_value_increment_warning(caplog):
    reset_suggested_seed_values()
    initializer_name = "warn_test"
    set_suggested_seed_value(initializer_name, 5)
    with caplog.at_level(logging.WARNING):
        set_suggested_seed_value(initializer_name, 3)
    assert (
        "override this value" in caplog.text
    ), "Warning about incorrect seed sequence should be logged"


def test_negative_seed_value():
    with pytest.raises(ValueError, match="Seed value should be >= 1"):
        set_suggested_seed_value("negative_test", -1)


def test_zero_seed_value():
    with pytest.raises(ValueError, match="Seed value should be >= 1"):
        set_suggested_seed_value("negative_test", 0)
