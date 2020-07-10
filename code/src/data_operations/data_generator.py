import numpy as np
import pydicom
from pydicom.data import get_testdata_files
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize

import config

# Will need to encode categories before calling the Data Generator
# Also before implement split for validation and train include shuffling

"""
INPUTS FOR DATAGENERATOR CLASS
list_IDs - array of strings of file paths to dcm file 
labels - encoded labels for the corresponding files
batch size
dimensions - as from config the dimension of input image to the model
n_channels - number of channels for image - (will be 1 for us in greyscale)
shuffle - shuffle starting position for each epoch
"""

"""
IMPLEMENTATION
Assuming csv file with available information loaded as a dataframe
- extract labels from dataframe
- extract paths from dataframe
- encode labels from dataframe
instantiate train_datagenerator and validation_datagenerator
run training using :
model.fit(x=training_generator,
=                    validation_data=validation_generator,
                    epochs = config.epochs,
                    callbacks=[
            EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=6)
        ]
"""


class DataGenerator(Sequence):
    """
    Generates data using Sequence to cycle through images to be processed for training
    """

    def __init__(self, list_IDs, labels, batch_size=config.BATCH_SIZE,
                 dim=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH']),
                 n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.labels = labels
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches to be run per epoch
        :return: Number of batches
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        :param index: point in samples
        :return: generates a batch
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        y = [self.labels[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X, np.array(y)

    def on_epoch_end(self):
        """
        Update the order of indexes at the very beginning and at the end of each epoch (so that batches between epochs
        do not look alike to get a more robust model).
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generate the data for a batch
        :param list_IDs_temp: ID's to be in the batch
        :return: images as arrays for the batch
        """
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # process dicom image into array and resize to input dimensions and add to batch
            X[i,] = load_dicom(ID, self.dim)

        return X


def load_dicom(path, dim):
    """
    Method
    :param path:
    :param dim:
    :return:
    """
    image_dicom = pydicom.dcmread(path)
    image_as_array = image_dicom.pixel_array
    resized_image = resize(image_as_array, (dim[0], dim[1], 1), anti_aliasing=True)
    return resized_image
