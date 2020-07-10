from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


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
