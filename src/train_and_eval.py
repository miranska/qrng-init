import logging
from types import MappingProxyType
from datapipeline import load_cifar10, load_mnist, load_imdb_reviews
from global_config import reset_suggested_seed_values
from models import ModelBuilder
from trainer import train_and_evaluate_model
import keras
import platform
import argparse
import os
import json
import random

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def reset_state(starting_seed_value: int) -> None:
    """
    Reset state after every iteration: reduce memory consumption,
    reset seeds.

    :param starting_seed_value: starting seed value
    """
    keras.backend.clear_session()
    reset_suggested_seed_values(starting_seed_value=starting_seed_value)


def get_keras_backend():
    # Check if the environment variable 'KERAS_BACKEND' exists
    keras_backend = os.getenv("KERAS_BACKEND")
    if keras_backend:
        return keras_backend

    # Path to the keras.json configuration file
    keras_config_path = os.path.expanduser("~/.keras/keras.json")

    # Check if keras.json exists and read the 'backend' attribute
    if os.path.isfile(keras_config_path):
        try:
            with open(keras_config_path, "r") as file:
                config = json.load(file)
            # Return the 'backend' attribute if it exists in the config
            return config.get("backend", "unknown")
        except Exception:
            return "unknown"
    else:
        return "unknown"


def int_or_none(value):
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a series of experiments on a given dataset using a "
            "specified deep learning model configuration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=None,
        help="Id of experiment, should be none or >= 1.",
    )
    parser.add_argument(
        "--sequence_scheme",
        type=str,
        default=None,
        help="Seed selection scheme for quasi-random initializer. "
        "Either 'auto', 'increment-start', or 'increment-end'. "
        "'auto' will pick the seed that leads to the best accuracy on "
        "the test set after the first epoch. "
        "Currently, we randomly select 5 seeds in the range [0, 10]."
        "`increment-start` which increments starting seed by 1. "
        "For example, if a model needs three dimensions, "
        "first experiment will get '1, 2, 3'; the second one '2, 3, 4', "
        "and so on. `increment-end` which increments starting seed based "
        "on the number of seeds used in the previous experiment. "
        "For example, if a model needs three dimensions, "
        "first experiment will get '1, 2, 3'; the second one '4, 5, 6'', "
        "and so on.",
    )
    parser.add_argument(
        "--auto_seeds_count",
        type=int,
        default=5,
        help="The number of seeds to try in the auto seed selection scheme.",
    )
    parser.add_argument(
        "--auto_min_seed_value",
        type=int,
        default=1,
        help="The minimum value of the seed to try "
        "in the auto seed selection scheme, should be none or >= 1.",
    )
    parser.add_argument(
        "--auto_max_seed_value",
        type=int,
        default=10,
        help="The maximum value of the seed to try "
        "in the auto seed selection scheme.",
    )
    parser.add_argument(
        "--auto_epoch_count",
        type=int,
        default=1,
        help="The number of epochs to train a model "
        "in the auto seed selection scheme.",
    )
    parser.add_argument(
        "--auto_repetition_count",
        type=int,
        default=1,
        help="The number of times to train a model with a particular seed "
        "in the auto seed selection scheme. The value should be >= 1.",
    )
    parser.add_argument(
        "--min_dim_id",
        type=int,
        default=1,
        help="Smallest dimension to use. The absolute min value is 1. "
        "In the original draft of the paper we used 2. The parameter is used "
        "only in 'increment-start' or 'increment-end' schemas.",
    )

    parser.add_argument(
        "--dataset_id",
        type=str,
        default="cifar10",
        help=(
            "Identifier for the dataset to use, either "
            "'mnist', 'cifar10', or 'imdb_reviews'."
        ),
    )
    parser.add_argument(
        "--kernel_initializer",
        type=str,
        default="glorot-uniform",
        help="The initializer for the kernel weights. "
        "Choose on of the following: "
        "'glorot-normal', 'glorot-uniform', 'he-normal', "
        "'he-uniform', 'lecun-normal', 'lecun-uniform', "
        "'orthogonal', 'random-normal', 'random-uniform', "
        "or 'truncated-normal'.",
    )
    parser.add_argument(
        "--initializer_type",
        type=str,
        default="quasi-random",
        help=(
            "Type of initializer to use, either 'quasi-random'"
            " or 'pseudo-random'."
        ),
    )
    parser.add_argument(
        "--units",
        type=int,
        default=None,
        help=(
            "Number of units for ANN and LSTM models, "
            "number of filters in the first convolutional layer for CNN, "
            "number of units in the feedforward network for Transformer."
        ),
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help=(
            "Word embedding dimensionality "
            "required for LSTM and Transformer models."
        ),
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="baseline-cnn",
        help="Identifier for the model configuration."
        " Choose one of the following: 'baseline-ann', "
        "'baseline-cnn', 'baseline-lstm', or "
        "'baseline-transformer'.",
    )
    parser.add_argument(
        "--max_features",
        type=int_or_none,
        default=None,
        help="Maximum vocabulary size, required for imbd_reviews dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of samples per batch of computation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results.json",
        help="Path to save the results of the experiment.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help=(
            "Optimizer to use for training the model. Choose one of "
            "the following: 'adam' or 'sgd'."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    return parser.parse_args(args)


def check_config(
    initializer_type,
    dataset_id,
    model_id,
    units,
    embedding_dim,
    experiment_id,
    sequence_scheme,
):
    if (dataset_id == "cifar10" or dataset_id == "mnist") and (
        model_id == "baseline-lstm" or model_id == "baseline-transformer"
    ):
        raise ValueError(
            "For 'cifar10' and 'mnist' datasets, 'model_id' cannot be "
            "'baseline-lstm' or 'baseline-transformer'."
        )
    elif (
        dataset_id == "imdb_reviews"
        and model_id != "baseline-lstm"
        and model_id != "baseline-transformer"
    ):
        raise ValueError(
            "For 'imdb_reviews' dataset, 'model_id' must be 'baseline-lstm'"
            " or 'baseline-transformer'."
        )
    if not units:
        raise ValueError("For all models, 'units' must be specified.")
    if (
        model_id in ("baseline-lstm", "baseline-transformer")
        and not embedding_dim
    ):
        raise ValueError(
            "Specify the embedding dimension for LSTM and Transformer models."
        )
    if initializer_type == "quasi-random":
        if sequence_scheme not in ("auto", "increment-start", "increment-end"):
            raise ValueError(
                "For 'quasi-random', 'sequence_scheme' must be "
                "'auto', 'increment-start', or 'increment-end'."
            )

        if experiment_id is None or sequence_scheme is None:
            raise ValueError(
                "Both, experiment_id and sequence_scheme must be specified."
            )


def prep_dataset(dataset_id, max_features, model_id):
    if dataset_id == "cifar10":
        if model_id.startswith("baseline-ann"):
            flatten = True
            input_shape = [32 * 32 * 3]
        else:
            flatten = False
            input_shape = [32, 32, 3]
        train_dg, val_dg, test_dg = load_cifar10(flatten=flatten)
        output_shape = 10
    elif dataset_id == "mnist":
        if model_id.startswith("baseline-ann"):
            flatten = True
            input_shape = [28 * 28 * 1]
        else:
            flatten = False
            input_shape = [28, 28, 1]

        train_dg, val_dg, test_dg = load_mnist(flatten=flatten)
        output_shape = 10
    elif dataset_id == "imdb_reviews":
        if not max_features:
            raise ValueError(
                "For 'imdb_reviews' dataset, 'max_features' "
                "must be specified."
            )
        train_dg, val_dg, test_dg = load_imdb_reviews(
            max_features=max_features
        )
        input_shape = [512]
        output_shape = 2
    else:
        raise ValueError(
            "Invalid dataset_id. Must be 'cifar10', 'mnist', "
            "or 'imdb_reviews'."
        )
    return input_shape, output_shape, test_dg, train_dg, val_dg


def get_starting_dim_id(
    experiment_id: int,
    model_id: str,
    sequence_scheme: str,
    min_dim_id: int = 1,
) -> int:
    """
    Auto-select starting dimension id for Sobol' sequences.
    We have two schemes:

    `increment-start` which increments starting seed by 1.
    For example, if a model needs three dimensions,
    first experiment will get "1, 2, 3"; the second one "2, 3, 4", and so on.

    `increment-end` which increments starting seed based on the number of
    seeds used in the previous experiment.
    For example, if a model needs three dimensions,
    first experiment will get "1, 2, 3"; the second one "4, 5, 6", and so on.


    :param experiment_id: Experiment id, should be >= 1.
    :param model_id: Identifier for the model configuration.
    :param sequence_scheme: Either "increment-start" or "increment-end".
    :param min_dim_id: the smallest id to use.
           In the first draft ot the paper we had `min_dim_id=2`.
    :return: starting seed
    """

    if sequence_scheme == "increment-start":
        return experiment_id - 1 + min_dim_id
    elif sequence_scheme == "increment-end":
        # delta is the number of dimensions used by a given model_id
        if model_id == "baseline-ann":
            delta = 3
        elif model_id == "baseline-cnn":
            delta = 4
        elif model_id == "baseline-lstm":
            delta = 2
        elif model_id == "baseline-transformer":
            delta = 7
        else:
            raise ValueError(f"Unknown model {model_id}")
        return (experiment_id - 1) * delta + min_dim_id
    else:
        raise ValueError(f"Unknown sequence scheme {sequence_scheme}")


def get_starting_dim_id_auto(
    epochs,
    max_seed_value,
    min_seed_value,
    seeds_count,
    repetition_count,
    build_cfg,
) -> int:
    """
    Search for the seed by trying a few values and
    taking a winner after `epochs` epoch.

    :param epochs: Number of epochs to train the model.
    :param max_seed_value: Maximum values of the seed to try,
           should be <= 21201.
    :param min_seed_value: Minimum values of the seed to try, should be >=1.
    :param seeds_count: Number of seeds to try.
    :param repetition_count: Number of times to train the model
           with the same seed.
    :param build_cfg: Parameters of `build_train_and_evaluate_model`.
    :return: Suggested seed_id
    """

    starting_dim_id = None
    best_accuracy = -1.0
    seeds = sorted(
        random.sample(
            range(min_seed_value, max_seed_value + 1),
            seeds_count,
        )
    )
    log.info(f"Choosing the best seed from this set: {seeds}")
    build_cfg_local = build_cfg.copy()
    build_cfg_local["epochs"] = epochs
    for seed in seeds:
        for repetition in range(repetition_count):
            reset_state(starting_seed_value=seed)
            accuracy = build_train_and_evaluate_model(
                print_model_summary=False, **build_cfg_local
            )["summary_metrics"]["test-categorical_accuracy"][0]
            log.debug(
                f"Current accuracy with seed {seed} and "
                f"repetition {repetition} is {accuracy}"
            )
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                starting_dim_id = seed
                log.debug(
                    f"New best seed: {starting_dim_id}; "
                    f"Best accuracy: {best_accuracy}"
                )
            else:
                log.debug("Keep seed as-is as accuracy did not improve")
    return starting_dim_id


def run_experiment(
    experiment_id,
    sequence_scheme,
    min_dim_id=1,
    dataset_id="cifar10",
    kernel_initializer="glorot-uniform",
    initializer_type="quasi-random",
    units=None,
    embedding_dim=None,
    model_id="baseline-cnn",
    max_features=None,
    batch_size=64,
    epochs=30,
    optimizer="adam",
    learning_rate=1e-4,
    auto_seeds_count=5,
    auto_min_seed_value=0,
    auto_max_seed_value=10,
    auto_epoch_count=1,
    auto_repetition_count=1,
):
    """
    Run a series of experiments on a given dataset using a specified deep
    learning model configuration.

    Args:
        experiment_id (int): ID of experiment, should be >= 1.
        sequence_scheme (str): Seed sequence scheme for quasi-random
             generator. Either "auto", "increment-start", or "increment-end".
        min_dim_id (int): Minimum dimension id to use in the quasi-random
             generator. Needed onty for "increment-start"
             or "increment-end" schemas.
        dataset_id (str): Identifier for the dataset to use, either
            "mnist", "cifar10", or "imdb_reviews".
        kernel_initializer (str): The initializer for the kernel weights.
        initializer_type (str): Type of initializer to use, either
            "quasi-random" or "pseudo-random".
        units (int, optional): Number of units in the dense or recurrent
            layers, required for LSTM models.
        embedding_dim (int, optional): Dimensionality of word embedding,
            required for LSTM and Transformer models.
        model_id (str): Identifier for the model configuration.
        max_features (int, optional): Maximum vocabulary size, required
            for imbd_reviews dataset.
        batch_size (int): Number of samples per batch of computation.
        epochs (int): Number of epochs to train the model.
        optimizer (str): Optimizer to use for training the model, choose one
            of the following: "adam" or "sgd".
        learning_rate (float): Learning rate for the optimizer.
        auto_seeds_count (int): The number of seeds to try
            in the auto seed selection scheme.
        auto_min_seed_value (int): The minimum value of the seed to try
            in the auto seed selection scheme.
        auto_max_seed_value (int): The maximum value of the seed to try
            in the auto seed selection scheme.
        auto_epoch_count (int): The number of epochs to try
            in the auto seed selection scheme.
        auto_repetition_count (int): Number of times to train the model
            with the same seed in the auto seed selection scheme.

    Returns:
        list: A list containing the training statistics for each iteration
        of the experiment. Each entry is a dictionary that details the
        performance metrics of the model across training, validation,
        and testing phases.

    This function handles the entire process of loading data, building the
    model with the specified configuration, training, and evaluating it over
    a given number of iterations. The results of each iteration are logged
    and returned for further analysis.
    """
    check_config(
        initializer_type,
        dataset_id,
        model_id,
        units,
        embedding_dim,
        experiment_id,
        sequence_scheme,
    )

    input_shape, output_shape, test_dg, train_dg, val_dg = prep_dataset(
        dataset_id, max_features, model_id
    )

    # use MappingProxyType to make the dictionary read-only
    model_config = MappingProxyType(
        {
            "initializer_type": initializer_type,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": "zeros",
            "recurrent_initializer": "orthogonal",  # used only by LSTM
            "max_features": max_features,  # for IMDB dataset
            "units": units,  # Used by LSTM and 2-layered ANN
            "embedding_dim": embedding_dim,  # Used by LSTM and Transformer
        }
    )

    # use MappingProxyType to make the dictionary read-only
    build_config = MappingProxyType(
        {
            "batch_size": batch_size,
            "input_shape": input_shape,
            "learning_rate": learning_rate,
            "model_config": model_config,
            "model_id": model_id,
            "optimizer": optimizer,
            "output_shape": output_shape,
            "test_dg": test_dg,
            "train_dg": train_dg,
            "val_dg": val_dg,
        }
    )

    starting_dim_id = None
    if initializer_type == "quasi-random":
        if sequence_scheme != "auto":
            starting_dim_id = get_starting_dim_id(
                experiment_id=experiment_id,
                model_id=model_id,
                sequence_scheme=sequence_scheme,
                min_dim_id=min_dim_id,
            )
        else:
            starting_dim_id = get_starting_dim_id_auto(
                max_seed_value=auto_max_seed_value,
                min_seed_value=auto_min_seed_value,
                seeds_count=auto_seeds_count,
                epochs=auto_epoch_count,
                repetition_count=auto_repetition_count,
                build_cfg=build_config,
            )

        log.info(
            f"Resetting the state. "
            f"Starting QRNG dimension is now set to {starting_dim_id}."
        )
        reset_state(starting_seed_value=starting_dim_id)
    elif initializer_type == "pseudo-random":
        sequence_scheme = "random"
        log.info(
            "Seeds for this experiment will be selected at random. "
            f"Setting sequence_scheme to {sequence_scheme}."
        )

    # Run experiment
    log.info(f"Running experiment #{experiment_id}")

    cur_stats = build_train_and_evaluate_model(
        print_model_summary=True,
        epochs=epochs,
        **build_config,
    )

    # append variables of interest to stats
    cur_stats.update(
        {
            "dataset": dataset_id,
            "initializer": kernel_initializer,
            "initializer_type": model_config["initializer_type"],
            "model": model_id,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "bias_initializer": model_config["bias_initializer"],
            "recurrent_initializer": model_config["recurrent_initializer"],
            "max_features": model_config["max_features"],
            "units": model_config["units"],
            "embedding_dim": model_config["embedding_dim"],
            "compute_node": platform.node(),
            "backend": get_keras_backend(),
            "sequence_scheme": sequence_scheme,
            "starting_dim_id": (
                starting_dim_id
                if initializer_type == "quasi-random"
                else -1  # as it is meaningless for pseudo-random
            ),
            "auto_seeds_count": auto_seeds_count,
            "auto_min_seed_value": auto_min_seed_value,
            "auto_max_seed_value": auto_max_seed_value,
            "auto_epoch_count": auto_epoch_count,
            "auto_repetition_count": auto_repetition_count,
        }
    )

    return [cur_stats]


def build_train_and_evaluate_model(
    batch_size,
    epochs,
    input_shape,
    learning_rate,
    model_config,
    model_id,
    optimizer,
    output_shape,
    test_dg,
    train_dg,
    val_dg,
    print_model_summary,
):
    model_builder = ModelBuilder(
        model_id=model_id,
        input_shape=input_shape,
        output_shape=output_shape,
        output_activation="softmax",
        model_config=model_config,
    )
    model = model_builder.get_model()

    if print_model_summary:
        model.summary()

    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid optimizer. Must be 'adam' or 'sgd'.")
    cur_stats = train_and_evaluate_model(
        model=model,
        optimizer=opt,
        loss_fn=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
        train_dg=train_dg,
        val_dg=val_dg,
        test_dg=test_dg,
        batch_size=batch_size,
        epochs=epochs,
    )

    return cur_stats


if __name__ == "__main__":
    my_args = parse_args()
    stats_out_path = my_args.out_path
    delattr(my_args, "out_path")
    my_stats = run_experiment(**vars(my_args))
    with open(stats_out_path, "w") as f:
        json.dump(my_stats, f)
    log.info(f"Results saved to {stats_out_path}")
