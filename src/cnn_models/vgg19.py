import ssl

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import config

# Required to download pre-trained weights for ImageNet (stored in ~/.keras/models/)
ssl._create_default_https_context = ssl._create_unverified_context


def create_vgg19_model(num_classes: int):
    """
    Creates a CNN from an existing architecture with pre-trained weights on ImageNet.
    :return: The VGG19 model.
    """
    base_model = Sequential(name="Base_Model")

    # Reconfigure a single channel image input (greyscale) into a 3-channel greyscale input (tensor).
    single_channel_input = Input(shape=(config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE['WIDTH'], 1))
    triple_channel_input = Concatenate()([single_channel_input, single_channel_input, single_channel_input])
    input_model = Model(inputs=single_channel_input, outputs=triple_channel_input)
    base_model.add(input_model)

    # Generate extra convolutional layers for model to put at the beginning
    base_model.add(Conv2D(64, (5, 5),
                          activation='relu',
                          padding='same'))
    base_model.add(Conv2D(32, (5, 5),
                          activation='relu',
                          padding='same'))
    base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluding the fully
    # connected layers.
    base_model.add(Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same'))
    pre_trained_model = VGG19(include_top=False, weights="imagenet",
                              input_shape=[config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 3])

    # Exclude input layer and first convolutional layer of VGG model.
    pre_trained_model_trimmed = Sequential(name="Pre-trained_Model")
    for layer in pre_trained_model.layers[2:]:
        pre_trained_model_trimmed.add(layer)

    # Add fully connected layers
    model = Sequential(name="Breast_Cancer_Model")

    # Start with base model consisting of convolutional layers
    model.add(base_model)
    model.add(pre_trained_model_trimmed)

    # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
    model.add(Flatten())

    # Add fully connected hidden layers and dropout layers between each for regularisation.
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation='relu', kernel_initializer="random_uniform", name='Dense_1'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu', kernel_initializer="random_uniform", name='Dense_2'))

    # Final output layer that uses softmax activation function (because the classes are exclusive).
    if config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
        model.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
    elif config.dataset == "mini-MIAS":
        model.add(Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", ame='Output'))

    # Print model details if running in debug mode.
    if config.verbose_mode:
        print(base_model.summary())
        print(pre_trained_model_trimmed.summary())
        print(model.summary())

    return model
