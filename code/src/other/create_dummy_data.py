from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from skimage.transform import resize


def get_fashion_images():
    """
    Function to load mnist_fashion data set:
        * Only using a subset of the data for testing functionality of the model
        * Resize the images to 512*512
    :return: training and testing data
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[0:600]
    y_train = y_train[0:600]
    x_test = x_test[0:100]
    y_test = y_test[0:100]

    x_train_new = []
    x_test_new = []
    for pic in x_train:
        upgraded_pic = resize(pic, (512, 512), anti_aliasing=True)
        x_train_new.append(upgraded_pic)
    x_train_new = np.array(x_train_new)
    for pic in x_test:
        upgraded_pic = resize(pic, (512, 512), anti_aliasing=True)
        x_test_new.append(upgraded_pic)
    x_test_new = np.array(x_test_new)

    # Encode labels via one-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_train = y_train.reshape(len(y_train), 1)
    integer_encoded_test = y_test.reshape(len(y_test), 1)

    y_train = onehot_encoder.fit_transform(integer_encoded_train)
    y_test = onehot_encoder.fit_transform(integer_encoded_test)

    return x_train_new, x_test_new, y_train, y_test


def generate_all_data():
    """
    Function to generate all dummy training validation and testing data for functionality testing of network
    :return: training, validation, testing data and imageDataGenerators
    """
    trainX, testX, trainY, testY = get_fashion_images()

    # Split trianing into training and validation
    trainX, valX, trainY, valY = train_test_split(trainX, trainY,
                                                      test_size=0.25, stratify=trainY, random_state=42)

    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    # initialize the validation/testing data augmentation object
    valAug = ImageDataGenerator()

    return trainX, trainY, valX, valY, testX, testY, trainAug, valAug
