import random

import numpy as np
import skimage as sk
import skimage.transform

import config


def generate_image_transforms(images, labels):
    """
    Oversample data by transforming existing images.
    :param images: input images
    :param labels: input labels
    :return: updated list of images and labels with extra transformed images and labels
    """
    images_with_transforms = images
    labels_with_transforms = labels

    available_transforms = {'rotate': random_rotation,
                            'noise': random_noise,
                            'horizontal_flip': horizontal_flip}

    class_balance = get_class_balances(labels)
    max_count = max(class_balance)
    to_add = [max_count - i for i in class_balance]

    for i in range(len(to_add)):
        if int(to_add[i]) == 0:
            continue
        label = np.zeros(len(to_add))
        label[i] = 1
        indices = [j for j, x in enumerate(labels) if np.array_equal(x, label)]
        indiv_class_images = [images[j] for j in indices]

        for k in range(int(to_add[i])):
            a = create_individual_transform(indiv_class_images[k % len(indiv_class_images)], available_transforms)
            transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)],
                                                            available_transforms)
            if config.model == "VGG":
                height = config.VGG_IMG_SIZE['HEIGHT']
                width = config.VGG_IMG_SIZE['WIDTH']
            elif config.model == "ResNet":
                height = config.RESNET_IMG_SIZE['HEIGHT']
                width = config.RESNET_IMG_SIZE['WIDTH']
            elif config.model == "Inception":
                height = config.INCEPTION_IMG_SIZE['HEIGHT']
                width = config.INCEPTION_IMG_SIZE['WIDTH']
            elif config.model == "Xception":
                height = config.XCEPTION_IMG_SIZE['HEIGHT']
                width = config.XCEPTION_IMG_SIZE['WIDTH']
            transformed_image = transformed_image.reshape(1, height, width, 1)

            images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
            transformed_label = label.reshape(1, len(label))
            labels_with_transforms = np.append(labels_with_transforms, transformed_label, axis=0)

    return images_with_transforms, labels_with_transforms


def random_rotation(image_array: np.ndarray):
    """
    Randomly rotate the image
    :param image_array: input image
    :return: randomly rotated image
    """
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: np.ndarray):
    """
    Add random noise to image
    :param image_array: input image
    :return: image with added random noise
    """
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: np.ndarray):
    """
    Flip image
    :param image_array: input image
    :return: horizantally flipped image
    """
    return image_array[:, ::-1]


def create_individual_transform(image: np.array, transforms: dict):
    """
    Create transformation of an individual image
    :param image: input image
    :param transforms: the possible transforms to do on the image
    :return: transformed image
    """
    num_transformations_to_apply = random.randint(1, len(transforms))
    num_transforms = 0
    transformed_image = None
    while num_transforms <= num_transformations_to_apply:
        key = random.choice(list(transforms))
        transformed_image = transforms[key](image)
        num_transforms += 1

    return transformed_image


def get_class_balances(y_vals):
    """
    Count occurrences of each class.
    :param y_vals: labels
    :return: array count of each class
    """
    num_classes = len(y_vals[0])
    counts = np.zeros(num_classes)
    for y_val in y_vals:
        for i in range(num_classes):
            counts[i] += y_val[i]
    return (counts.tolist())
