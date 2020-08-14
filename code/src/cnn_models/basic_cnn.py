from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import config


def create_basic_cnn_model(num_classes: int):
    """
    Function to create a basic CNN.
    :param num_classes: The number of classes (labels).
    :return: A basic CNN model.
    """
    model = Sequential()

    # Convolutional + spooling layers
    model.add(Conv2D(64, (5, 5), input_shape=(config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE['WIDTH'], 1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())

    # Dropout
    model.add(Dropout(0.5, seed=config.RANDOM_SEED, name="Dropout_1"))

    # FC
    model.add(Dense(1024, activation='relu', name='Dense_2'))

    # Output
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
    else:
        model.add(Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output'))

    # Print model details if running in debug mode.
    if config.verbose_mode:
        print(model.summary())

    return model
