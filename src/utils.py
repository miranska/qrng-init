import logging
import matplotlib.pyplot as plt
import keras
import numpy as np

log = logging.getLogger(__name__)


def split_data(x_train, y_train, training_observations_cnt):
    log.info("Splitting Data into Train and Validation")
    log.info(
        f"Training observation count is set to {training_observations_cnt}"
    )
    log.info(f"x_train shape before splitting {x_train.shape}")
    log.info(f"y_train shape before splitting {y_train.shape}")
    dat_offset = x_train.shape[0] - training_observations_cnt
    if training_observations_cnt == x_train.shape[0]:
        log.info(
            "Training observation count equals the length of x_train. "
            "No validation set will be created."
        )

        x_val = keras.ops.convert_to_tensor(np.empty((0,) + x_train.shape[1:]))
        y_val = keras.ops.convert_to_tensor(np.empty((0,) + y_train.shape[1:]))
    else:
        x_val = x_train[-dat_offset:]
        y_val = y_train[-dat_offset:]
        x_train = x_train[:-dat_offset]
        y_train = y_train[:-dat_offset]
    log.info(f"x_train shape after splitting {x_train.shape}")
    log.info(f"y_train shape after splitting {y_train.shape}")
    log.info(f"x_val shape after splitting {x_val.shape}")
    log.info(f"y_val shape after splitting {y_val.shape}")

    return (x_train, y_train), (x_val, y_val)


def plot_demo(stats, epochs):
    # Function to load data
    def load_data(x):
        return [
            trial["summary_metrics"]["test-categorical_accuracy"]
            for trial in x
        ]

    # Load accuracies from all files with names
    all_accuracies_named = {
        name: load_data(path) for name, path in stats.items()
    }

    # Prepare boxplot data for each set of trials with names
    all_boxplot_data_named = {
        name: [[acc[i] for acc in accuracies] for i in range(epochs)]
        for name, accuracies in all_accuracies_named.items()
    }

    # Determine y-axis limits for better comparison using the named datasets
    y_limits_named = [
        (
            name.split()[0],
            (
                min(min(data) for data in dataset),
                max(max(data) for data in dataset),
            ),
        )
        for name, dataset in all_boxplot_data_named.items()
    ]
    y_limits = {}
    for k, v in y_limits_named:
        y_limits.setdefault(k, []).append(v)

    # Plotting with names
    _, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for _, (name, ax) in enumerate(zip(stats.keys(), axes.flat)):
        ax.boxplot(all_boxplot_data_named[name], positions=range(epochs))
        ax.set_title(name)
        ax.set_xticks(range(1, epochs + 1, 1))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Cat. Accuracy")
        ax.grid(True)

        # Set y-limits by row
        model = name.split()[0]
        cur_y_limits = (
            min(*[low[0] for low in y_limits[model]]),
            max(*[high[1] for high in y_limits[model]]),
        )
        ax.set_ylim(cur_y_limits)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.show()
