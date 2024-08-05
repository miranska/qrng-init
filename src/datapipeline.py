import logging
import numpy as np
import keras
import tensorflow as tf
from utils import split_data
from keras import ops

log = logging.getLogger(__name__)


def load_mnist(flatten=False, one_hot=True, training_observations_cnt=60000):
    log.info(
        f"Loading MNIST dataset with flatten={flatten} and one_hot={one_hot}"
    )
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if one_hot:
        log.info("Converting labels to one-hot encoding")
        y_train = ops.one_hot(y_train, num_classes=10)
        y_test = ops.one_hot(y_test, num_classes=10)

    if flatten:
        log.info("Flattening images")
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
    log.info("MNIST Dataset loaded")
    log.info(f"x_train shape {x_train.shape}")
    log.info(f"y_train shape {y_train.shape}")
    log.info(f"x_test shape {x_test.shape}")
    log.info(f"y_test shape {y_test.shape}")

    (x_train, y_train), (x_val, y_val) = split_data(
        x_train, y_train, training_observations_cnt
    )

    traindg = create_datapipeline(x_train, y_train)
    valdg = create_datapipeline(x_val, y_val)
    testdg = create_datapipeline(x_test, y_test)

    return traindg, valdg, testdg


def load_cifar10(flatten=False, one_hot=True, training_observations_cnt=50000):
    log.info(
        f"Loading CIFAR10 dataset with flatten={flatten} and one_hot={one_hot}"
    )
    (x_train, y_train), (
        x_test,
        y_test,
    ) = keras.datasets.cifar10.load_data()

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    if one_hot:
        log.info("Converting labels to one-hot encoding")
        y_train = ops.one_hot(y_train, num_classes=10)
        y_test = ops.one_hot(y_test, num_classes=10)

    if flatten:
        log.info("Flattening images")
        x_train = np.reshape(x_train, (-1, 3072))
        x_test = np.reshape(x_test, (-1, 3072))

    log.info("CIFAR10 Dataset loaded")
    log.info(f"x_train shape {x_train.shape}")
    log.info(f"y_train shape {y_train.shape}")
    log.info(f"x_test shape {x_test.shape}")
    log.info(f"y_test shape {y_test.shape}")

    (x_train, y_train), (x_val, y_val) = split_data(
        x_train, y_train, training_observations_cnt
    )

    traindg = create_datapipeline(x_train, y_train)
    valdg = create_datapipeline(x_val, y_val)
    testdg = create_datapipeline(x_test, y_test)

    return traindg, valdg, testdg


def load_imdb_reviews(
    one_hot=True,
    max_features=20000,
    seq_len=512,
    training_observations_cnt=25000,
):
    log.info("Loading IMDB reviews dataset")
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=max_features,
    )
    # Use pad_sequence to standardize sequence length
    x_train = keras.utils.pad_sequences(x_train, maxlen=seq_len)
    x_test = keras.utils.pad_sequences(x_test, maxlen=seq_len)

    if one_hot:
        log.info("Converting labels to one-hot encoding")
        y_train = ops.one_hot(y_train, num_classes=2)
        y_test = ops.one_hot(y_test, num_classes=2)
    log.info("IMDB reviews dataset loaded")
    log.info(f"x_train shape {x_train.shape}")
    log.info(f"y_train shape {y_train.shape}")
    log.info(f"x_test shape {x_test.shape}")
    log.info(f"y_test shape {y_test.shape}")

    (x_train, y_train), (x_val, y_val) = split_data(
        x_train, y_train, training_observations_cnt
    )

    traindg = create_datapipeline(x_train, y_train)
    valdg = create_datapipeline(x_val, y_val)
    testdg = create_datapipeline(x_test, y_test)

    return traindg, valdg, testdg


def create_datapipeline(X, Y, batch_size=64):
    datapipeline = tf.data.Dataset.from_tensor_slices((X, Y))
    datapipeline = datapipeline.shuffle(
        buffer_size=1024, reshuffle_each_iteration=True
    )
    datapipeline = datapipeline.batch(batch_size=batch_size)
    datapipeline = datapipeline.prefetch(1)
    return datapipeline
