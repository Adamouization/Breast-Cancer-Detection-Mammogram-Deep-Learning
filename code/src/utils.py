import matplotlib.pyplot as plt
from numpy.random import seed
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

import config


def set_random_seeds() -> None:
    """
    Set random seeds for reproducible results.
    :return: None.
    """
    seed(config.RANDOM_SEED)  # NumPy
    tf.random.set_seed(config.RANDOM_SEED)  # Tensorflow


def print_runtime(text: str, runtime: float) -> None:
    """
    Print runtime in seconds.
    :param text: Message to print to the console indicating what was measured.
    :param runtime: The runtime in seconds.
    :return: None.
    """
    print("\n--- {} runtime: {} seconds ---".format(text, runtime))


def show_raw_image(img) -> None:
    """
    Displays a PIL image.
    :param img: the image in PIL format (before being converted to an array).
    :return: None.
    """
    img.show()


def print_num_gpus_available() -> None:
    """
    Prints the number of GPUs available on the current machine.
    :return: None
    """
    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def print_error_message() -> None:
    """
    Print error message and exit code when a CLI-related error occurs.
    :return:
    """
    print("Wrong command line arguments passed, please use 'python main.py --help' for instructions on which arguments"
          "to pass to the program.")
    exit(1)


def create_label_encoder():
    """
    Creates the label encoder.
    :return: The instantiated label encoder.
    """
    return LabelEncoder()


def print_cli_arguments() -> None:
    """
    Print command line arguments and all code configurations to the terminal.
    :return: None
    """
    print("\nSettings used:")
    print("Dataset: {}".format(config.dataset))
    print("Mammogram type: {}".format(config.mammogram_type))
    print("CNN Model: {}".format(config.model))
    print("Run mode: {}".format(config.run_mode))
    print("Learning rate: {}".format(config.learning_rate))
    print("Batch size: {}".format(config.batch_size))
    print("Max number of epochs when original CNN layers are frozen: {}".format(config.max_epoch_frozen))
    print("Max number of epochs when original CNN layers are unfrozen: {}".format(config.max_epoch_unfrozen))
    print("Verbose mode: {}".format(config.verbose_mode))
    print("Experiment name: {}\n".format(config.name))


def save_output_figure(title: str) -> None:
    """
    Save a figure on the output directory.
    :param title: The title of the figure.
    :return: None
    """
    plt.savefig(
        "../output/{}_dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_{}.png".format(
            config.run_mode,
            config.dataset,
            config.mammogram_type,
            config.model,
            config.learning_rate,
            config.batch_size,
            config.max_epoch_frozen,
            config.max_epoch_unfrozen,
            config.is_roi,
            config.name,
            title))  # bbox_inches='tight'


def load_trained_model() -> None:
    """
    Load the model previously trained for the final evaluation using the test set.
    :return: None
    """
    print("Loading trained model")
    return load_model(
        "/cs/scratch/agj6/saved_models/dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_saved-model.h5".format(
            config.dataset,
            config.mammogram_type,
            config.model,
            config.learning_rate,
            config.batch_size,
            config.max_epoch_frozen,
            config.max_epoch_unfrozen,
            config.is_roi,
            config.name)
    )
