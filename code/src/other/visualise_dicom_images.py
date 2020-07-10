import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
import tensorflow as tf
import tensorflow_io as tfio

import config
from data_operations.dataset_feed import create_dataset
from data_operations.data_preprocessing import dataset_stratified_split


def main() -> None:
    parse_command_line_arguments()

    path = "/cs/tmp/datasets/CBIS-DDSM/Calc-Training_P_00005_RIGHT_CC/08-07-2016-DDSM-23157/1.000000-full mammogram images-38548/1-1.dcm"
    dataset = pydicom.dcmread(path)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.savefig("../output/image_original.png")

    image = resize(dataset.pixel_array, [512, 512])
    imgplot = plt.imshow(image, cmap=plt.cm.bone)
    plt.savefig("../output/image_resized.png")

    # resize keep aspect ratio
    df = pd.read_csv("../../data/CBIS-DDSM/training.csv")
    list_IDs = df['img_path'].values
    labels = df['label'].values

    X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=list_IDs, labels=labels)

    dataset_train = create_dataset(X_train, y_train)

    label = labels[0]
    list_ID = list_IDs[0]

    image = parse_function(list_ID, label)
    plt.imshow(np.squeeze(image[0].numpy()), cmap='gray')
    plt.savefig("../output/image_maintained_AR.png")


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their value in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default="mini-MIAS",
                        required=True,
                        help="The dataset to use. Must be either 'mini-MIAS' or 'CBIS-DDMS'."
                        )
    parser.add_argument("-m", "--model",
                        default="basic",
                        required=True,
                        help="The model to use. Must be either 'basic' or 'advanced'."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    args = parser.parse_args()
    config.dataset = args.dataset
    config.model = args.model
    config.verbose_mode = args.verbose


def parse_function(filename, label):
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize_with_pad(decoded_png, 512, 512)
    image /= 255

    return image, label


if __name__ == '__main__':
    main()
