import ssl

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def generate_vgg_model(classes_len: int):
    """
    Function to create a VGG19 model pre-trained with custom FC Layers.
    If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
    larger images.
    :param classes_len: The number of classes (labels).
    :return: The VGG19 model.
    """
    # Reconfigure single channel input into a greyscale 3 channel input
    img_input = Input(shape=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 1))
    img_conc = Concatenate()([img_input, img_input, img_input])

    # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
    model_base = VGG19(include_top=False, weights='imagenet', input_tensor=img_conc)

    # Add fully connected layers
    model = Sequential()
    # Start with base model consisting of convolutional layers
    model.add(model_base)

    # Generate additional convolutional layers
    if config.model == "advanced":
        model.add(Conv2D(1024, (3, 3),
                         activation='relu',
                         padding='same'))
        model.add(Conv2D(1024, (3, 3),
                         activation='relu',
                         padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
    model.add(Flatten())

    # Add fully connected hidden layers.
    model.add(Dense(units=512, activation='relu', name='Dense_Intermediate_1'))
    model.add(Dense(units=32, activation='relu', name='Dense_Intermediate_2'))

    # Possible dropout for regularisation can be added later and experimented with:
    # model.add(Dropout(0.1, name='Dropout_Regularization'))

    # Final output layer that uses softmax activation function (because the classes are exclusive).
    if classes_len == 2:
        model.add(Dense(1, activation='sigmoid', name='Output'))
    else:
        model.add(Dense(classes_len, activation='softmax', name='Output'))

    # Print model details if running in debug mode.
    if config.verbose_mode:
        print(model.summary())

    return model
