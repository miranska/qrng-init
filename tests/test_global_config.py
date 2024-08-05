import unittest
from unittest.mock import patch
from global_config import (
    get_suggested_seed_value,
    set_suggested_seed_value,
    reset_suggested_seed_values,
)


class TestSeedFunctions(unittest.TestCase):
    def setUp(self):
        # Reset the state before each test
        reset_suggested_seed_values()

    def test_get_suggested_seed_value(self):
        seed_value = get_suggested_seed_value("initializer_1")
        self.assertEqual(1, seed_value)

        seed_value = get_suggested_seed_value("initializer_2")
        self.assertEqual(1, seed_value)

        seed_value = get_suggested_seed_value("initializer_1")
        self.assertEqual(2, seed_value)

    def test_set_suggested_seed_value(self):
        initializer_name = "initializer_1"
        seed_value = 1

        with patch("global_config.log.warning") as mock_warning:
            set_suggested_seed_value(initializer_name, seed_value)
            self.assertEqual(
                seed_value + 1, get_suggested_seed_value(initializer_name)
            )
            mock_warning.assert_not_called()

    def test_override(self):
        # Test setting a seed value smaller and larger than the current value
        initializer_name = "initializer_1"
        current_value = 10
        set_suggested_seed_value(initializer_name, current_value)

        with patch("global_config.log.warning") as mock_warning:
            set_suggested_seed_value(initializer_name, 5)
            expected_seed_value = current_value
            mock_warning.assert_called_once_with(
                f"The latest seed value for {initializer_name} is "
                f"{current_value}. "
                f"You are trying to set the value to {5}. "
                f"Let's override this value with {expected_seed_value}."
            )
            self.assertEqual(
                11,
                get_suggested_seed_value(initializer_name),
            )

        with patch("global_config.log.warning") as mock_warning:
            set_suggested_seed_value(initializer_name, 15)
            expected_seed_value = current_value
            mock_warning.assert_called_once_with(
                f"The latest seed value for {initializer_name} is "
                f"{current_value+1}. "
                f"You are trying to set the value to {15}. "
                f"Let's override this value with {current_value+1}."
            )
            self.assertEqual(12, get_suggested_seed_value(initializer_name))

        # Test setting a negative seed value
        with self.assertRaises(ValueError):
            set_suggested_seed_value(initializer_name, -1)

        # Test setting a seed value different from 1 when not in the dictionary
        initializer_name = "initializer_3"
        seed_value = 5

        with patch("global_config.log.warning") as mock_warning:
            set_suggested_seed_value(initializer_name, seed_value)
            mock_warning.assert_called_once_with(
                f"You are starting the seed sequence from {initializer_name} "
                f"from {seed_value} rather than 1. "
                f"Are you sure this is what you want?"
            )
            self.assertEqual(6, get_suggested_seed_value(initializer_name))


if __name__ == "__main__":
    unittest.main()
