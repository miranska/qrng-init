import time
import logging

import keras.src.initializers as ki
from keras.src.callbacks import Callback

import custom_initializers as ci


log = logging.getLogger(__name__)


def get_initializer(
    initializer_id, initializer_type="pseudo-random", seed=None, **kwargs
):
    log.info(
        f"Initializing {initializer_id} from {initializer_type}, seed={seed}"
    )
    if initializer_type == "pseudo-random":
        if initializer_id == "glorot-normal":
            return ki.GlorotNormal
        elif initializer_id == "glorot-uniform":
            return ki.GlorotUniform
        elif initializer_id == "he-normal":
            return ki.HeNormal
        elif initializer_id == "he-uniform":
            return ki.HeUniform
        elif initializer_id == "lecun-normal":
            return ki.LecunNormal
        elif initializer_id == "lecun-uniform":
            return ki.LecunUniform
        elif initializer_id == "orthogonal":
            return ki.OrthogonalInitializer
        elif initializer_id == "random-normal":
            return ki.RandomNormal
        elif initializer_id == "random-uniform":
            return ki.RandomUniform
        elif initializer_id == "truncated-normal":
            return ki.TruncatedNormal
        elif initializer_id == "zeros":
            return ki.Zeros
        else:
            raise ValueError(f"Unknown initializer_id {initializer_id}")
    elif initializer_type == "quasi-random":
        if initializer_id == "glorot-normal":
            return ci.GlorotNormalQR
        elif initializer_id == "glorot-uniform":
            return ci.GlorotUniformQR
        elif initializer_id == "he-normal":
            return ci.HeNormalQR
        elif initializer_id == "he-uniform":
            return ci.HeUniformQR
        elif initializer_id == "lecun-normal":
            return ci.LecunNormalQR
        elif initializer_id == "lecun-uniform":
            return ci.LecunUniformQR
        elif initializer_id == "orthogonal":
            return ci.OrthogonalInitializerQR
        elif initializer_id == "random-normal":
            return ci.RandomNormalQR
        elif initializer_id == "random-uniform":
            return ci.RandomUniformQR
        elif initializer_id == "truncated-normal":
            return ci.TruncatedNormalQR
        elif initializer_id == "zeros":
            return ki.Zeros
        else:
            raise ValueError(f"Unknown initializer_id {initializer_id}")
    else:
        raise ValueError(f"Unknown initializer_type {initializer_type}")


class PerEpochStatsCallback(Callback):
    """
    Compute per-epochs stats.

    The accuracy value for a particular dataset (train, val, or test)
    is set to -1 for each of the epochs if it is not passed into
    the constructor.
    """

    def __init__(self, train_data=None, val_data=None, test_data=None):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.time_per_epoch = []
        self.accuracy_dict = {
            "train-categorical_accuracy": [],
            "val-categorical_accuracy": [],
            "test-categorical_accuracy": [],
        }

    def get_time_per_epoch(self):
        return self.time_per_epoch

    def get_accuracies(self):
        return self.accuracy_dict

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        train_acc = (
            self.model.evaluate(self.train_data, verbose=0)[1]
            if self.train_data is not None
            else -1
        )
        val_acc = (
            self.model.evaluate(self.val_data, verbose=0)[1]
            if self.val_data is not None
            else -1
        )
        test_acc = (
            self.model.evaluate(self.test_data, verbose=0)[1]
            if self.test_data is not None
            else -1
        )

        self.accuracy_dict["train-categorical_accuracy"].append(train_acc)
        self.accuracy_dict["val-categorical_accuracy"].append(val_acc)
        self.accuracy_dict["test-categorical_accuracy"].append(test_acc)
        self.time_per_epoch.append(epoch_time)

        log.info(
            f"Epoch {epoch + 1}: train accuracy = {train_acc}, "
            f"val accuracy = {val_acc}, test accuracy = {test_acc}, "
            f"time per epoch = {epoch_time:.2f} seconds"
        )


def train_and_evaluate_model(
    model,
    optimizer,
    loss_fn,
    metrics,
    batch_size,
    train_dg,
    val_dg,
    test_dg,
    epochs,
):

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=False,  # set to True for debugging
    )

    # per_epoch_callback = PerEpochStatsCallback(
    #     train_data=train_dg, val_data=val_dg, test_data=test_dg
    # )

    per_epoch_callback = PerEpochStatsCallback(test_data=test_dg)

    start_time = time.time()
    model.fit(
        train_dg,
        validation_split=0.0,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,  # it's ignored when we use TF pipeline
        callbacks=[per_epoch_callback],
    )

    return {
        "overall_time": time.time() - start_time,
        "time_per_epoch": per_epoch_callback.get_time_per_epoch(),
        "batch_size": batch_size,
        "summary_metrics": per_epoch_callback.get_accuracies(),
    }
