import ssl

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Model

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def generate_vgg_model_large(classes_len: int):
    """
    Function to create a VGG19 model pre-trained with custom FC Layers at the start of the network plus optional layers at
    the end before the fully connected ones as well
    If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
    larger images.
    This model is a larger model that starts with two more sets of convolutional layers with less filters 
    :param classes_len: The number of classes (labels).
    :return: The VGG19 model.
    """
    
    
    model_base = Sequential()

    # Reconfigure single channel input into a greyscale 3 channel input
    img_input = Input(shape=(config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'], 1))
    img_conc = Concatenate()([img_input, img_input, img_input])
    input_model = Model(inputs=img_input, outputs=img_conc)

    # Generate extra convolutional layers for model to put at the beginning
    model_base.add(input_model)
    model_base.add(Conv2D(16, (3, 3),
                      activation='relu',
                      padding='same'))
    
    model_base.add(Conv2D(16, (3, 3),
                      activation='relu',
                      padding='same'))
    
    model_base.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model_base.add(Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same'))
    
    model_base.add(Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same'))
    
    model_base.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # To ensure model fits with vgg model, we can remove the first layer from the vgg model to replace with this
    model_base.add(Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same'))

    # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
    vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=[config.VGG_IMG_SIZE['HEIGHT'],
                                                                           config.VGG_IMG_SIZE['HEIGHT'],
                                                                           3])
    

    # Crop vgg model to exlude input layer and first convolutional layer
    vgg_model_cropped = Sequential()
    for layer in vgg_model.layers[2:]: # go through until last layer
        vgg_model_cropped.add(layer)

    # Combine the models
    combined_model = Sequential()
    combined_model.add(model_base)
    combined_model.add(vgg_model_cropped)

    
    # Add fully connected layers
    model = Sequential()
    # Start with base model consisting of convolutional layers
    model.add(combined_model)

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
        model.summary()
        

    return model
