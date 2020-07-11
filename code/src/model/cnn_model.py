import ssl

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


class CNN_Model:

    def __init__(self, model_name: str, num_classes: int):
        """
        Function to create a VGG19 model pre-trained with custom FC Layers.
        If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
        larger images.
        :param model_name: The CNN model to use.
        :param num_classes: The number of classes (labels).
        :return: The VGG19 model.
        """
        self._model = Sequential()
        self.num_classes = num_classes
        self.model_name = model_name

        self.create_model()

    def create_model(self):
        """

        """
        # Reconfigure a single channel image input (greyscale) into a 3-channel greyscale input.
        single_channel_input = Input(shape=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 1))
        triple_channel_input = Concatenate()([single_channel_input, single_channel_input, single_channel_input])

        # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
        if self.model_name == "VGG":
            base_model = VGG19(include_top=False, weights='imagenet', input_tensor=triple_channel_input)

        # Add fully connected layers
        self.model = Sequential()

        # Start with base model consisting of convolutional layers
        self.model.add(base_model)

        # Generate additional convolutional layers
        if config.model == "advanced":
            self.model.add(Conv2D(1024, (3, 3),
                                  activation='relu',
                                  padding='same'))
            self.model.add(Conv2D(1024, (3, 3),
                                  activation='relu',
                                  padding='same'))
            self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
        self.model.add(Flatten())

        # Add fully connected hidden layers.
        self.model.add(Dense(units=512, activation='relu', name='Dense_Intermediate_1'))
        self.model.add(Dense(units=32, activation='relu', name='Dense_Intermediate_2'))

        # Possible dropout for regularisation can be added later and experimented with:
        # model.add(Dropout(0.1, name='Dropout_Regularization'))

        # Final output layer that uses softmax activation function (because the classes are exclusive).
        if self.num_classes == 2:
            self.model.add(Dense(1, activation='sigmoid', name='Output'))
        else:
            self.model.add(Dense(self.num_classes, activation='softmax', name='Output'))

        # Print model details if running in debug mode.
        if config.verbose_mode:
            print(self.model.summary())

    @property
    def model(self):
        return self.model()

    @model.setter
    def model(self, value):
        pass
