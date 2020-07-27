import ssl

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def create_inceptionv3_model(num_classes: int):
    """
    Function to create an InceptionV3 model pre-trained with custom FC Layers.
    :param num_classes: The number of classes (labels).
    :return: The custom InceptionV3 model.
    """
    # Reconfigure single channel input into a greyscale 3 channel input
    img_input = Input(shape=(config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE['WIDTH'], 1))
    img_conc = Concatenate()([img_input, img_input, img_input])

    # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
    model_base = InceptionV3(include_top=False, weights='imagenet', input_tensor=img_conc)

    # Add fully connected layers
    model = Sequential()
    # Start with base model consisting of convolutional layers
    model.add(model_base)

    # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
    model.add(Flatten())
    
    # Dropout layer for regularisation
    model.add(Dropout(0.2, name="Dropout_Regularisation"))

    # Fully connected layers.
    model.add(Dense(units=512, activation='relu', name='Dense_Intermediate_1'))
    model.add(Dense(units=32, activation='relu', name='Dense_Intermediate_2'))

    # Final output layer that uses softmax activation function (because the classes are exclusive).
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid', name='Output'))
    else:
        model.add(Dense(num_classes, activation='softmax', name='Output'))

    # Print model details if running in debug mode.
    if config.verbose_mode:
        print(model.summary())

    return model
