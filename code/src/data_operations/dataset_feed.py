import tensorflow as tf
import tensorflow_io as tfio

import config


def create_dataset(x, y):
    """
    Generates a TF dataset for feeding in the data.
    :param x: X inputs - paths to images
    :param y: y values - labels for images
    :return: the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # map values from dicom image path to array
    if config.image_size == "small":
        dataset = dataset.map(parse_function_small, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(parse_function_large, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def parse_function_small(filename, label):
    """
    mapping function to convert filename to array of pixel values
    :param filename:
    :param label:
    :return:
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, scale="auto", dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    if self.model_name == "VGG":
        height = config.VGG_IMG_SIZE['HEIGHT']
        width = config.VGG_IMG_SIZE['WIDTH']
    elif self.model_name == "ResNet":
        height = config.RESNET_IMG_SIZE['HEIGHT']
        width = config.RESNET_IMG_SIZE['WIDTH']
    elif self.model_name == "Inception":
        height = config.INCEPTION_IMG_SIZE['HEIGHT']
        width = config.INCEPTION_IMG_SIZE['WIDTH']
    elif self.model_name == "Xception":
        height = config.XCEPTION_IMG_SIZE['HEIGHT']
        width = config.XCEPTION_IMG_SIZE['WIDTH']
    image = tf.image.resize(decoded_png, [height, width])
    image /= 255

    return image, label


def parse_function_large(filename, label):
    """
    mapping function to convert filename to array of pixel values for larger images we use resize with padding
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes,color_dim = True,  dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    if self.model_name == "VGG":
        height = config.VGG_IMG_SIZE['HEIGHT']
        width = config.VGG_IMG_SIZE['WIDTH']
    elif self.model_name == "ResNet":
        height = config.RESNET_IMG_SIZE['HEIGHT']
        width = config.RESNET_IMG_SIZE['WIDTH']
    elif self.model_name == "Inception":
        height = config.INCEPTION_IMG_SIZE['HEIGHT']
        width = config.INCEPTION_IMG_SIZE['WIDTH']
    elif self.model_name == "Xception":
        height = config.XCEPTION_IMG_SIZE['HEIGHT']
        width = config.XCEPTION_IMG_SIZE['WIDTH']
    image = tf.image.resize_with_pad(decoded_png, height, width)
    image /= 255

    return image, label
