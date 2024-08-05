import logging

log = logging.getLogger(__name__)
# expected variable name to track latest seed value for Quasi-random generator
_max_qr_seed_values = {}
_starting_qr_seed_value = 1


def reset_suggested_seed_values(starting_seed_value: int = 1) -> None:
    """
    Reset seed values for all initializers.
    """
    global _max_qr_seed_values
    _max_qr_seed_values = {}
    global _starting_qr_seed_value
    _starting_qr_seed_value = starting_seed_value


def get_suggested_seed_value(initializer_name: str) -> int:
    """
    Get suggested random seed value for a given initializer.

    :param initializer_name: name of the initializer
    :return: seed value
    """
    global _starting_qr_seed_value
    if initializer_name in _max_qr_seed_values:
        _max_qr_seed_values[initializer_name] += 1
    else:
        _max_qr_seed_values[initializer_name] = _starting_qr_seed_value

    return _max_qr_seed_values[initializer_name]


def set_suggested_seed_value(initializer_name: str, seed_value: int) -> None:
    """
    Set suggested seed value for a given initializer.

    :param initializer_name: name of the initializer
    :param seed_value: suggested seed value
    :return: None
    """
    if seed_value < 1:
        raise ValueError(f"Seed value should be >= 1, got {seed_value}")

    if initializer_name in _max_qr_seed_values:
        current_seed_value = _max_qr_seed_values[initializer_name]
        if current_seed_value != seed_value:
            new_seed_value = _max_qr_seed_values[initializer_name]
            log.warning(
                f"The latest seed value for {initializer_name} is "
                f"{_max_qr_seed_values[initializer_name]}. "
                f"You are trying to set the value to {seed_value}. "
                f"Let's override this value with {new_seed_value}."
            )
            seed_value = new_seed_value
    else:
        if seed_value != 1:
            log.warning(
                f"You are starting the seed sequence from {initializer_name} "
                f"from {seed_value} rather than 1. "
                f"Are you sure this is what you want?"
            )
    _max_qr_seed_values[initializer_name] = seed_value
