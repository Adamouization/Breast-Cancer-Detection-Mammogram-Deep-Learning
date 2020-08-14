import argparse
from collections import Counter
import time

import numpy as np

from cnn_models.cnn_model import CnnModel, test_model_evaluation
import config
from data_operations.dataset_feed import create_dataset
from data_operations.data_preprocessing import calculate_class_weights, dataset_stratified_split, \
    import_cbisddsm_testing_dataset, import_cbisddsm_training_dataset, import_minimias_dataset
from data_operations.data_transformations import generate_image_transforms
from utils import create_label_encoder, load_trained_model, print_cli_arguments, print_error_message, \
    print_num_gpus_available, print_runtime, set_random_seeds


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which dataset and model to use.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :return: None.
    """
    set_random_seeds()
    parse_command_line_arguments()
    print_num_gpus_available()

    # Create label encoder.
    l_e = create_label_encoder()

    # Run in training mode.
    if config.run_mode == "train":

        print("-- Training model --\n")

        # Start recording time.
        start_time = time.time()

        # Multi-class classification (mini-MIAS dataset)
        if config.dataset == "mini-MIAS":
            # Import entire dataset.
            images, labels = import_minimias_dataset(data_dir="../data/{}/images_processed".format(config.dataset),
                                                     label_encoder=l_e)

            # Split dataset into training/test/validation sets (80/20% split).
            X_train, X_test, y_train, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels)

            # Create CNN model and split training/validation set (80/20% split).
            model = CnnModel(config.model, l_e.classes_.size)
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25,
                                                                      dataset=X_train,
                                                                      labels=y_train)

            # Calculate class weights.
            class_weights = calculate_class_weights(y_train, l_e)

            # Data augmentation.
            y_train_before_data_aug = y_train
            X_train, y_train = generate_image_transforms(X_train, y_train)
            y_train_after_data_aug = y_train
            np.random.shuffle(y_train)

            if config.verbose_mode:
                print("Before data augmentation:")
                print(Counter(list(map(str, y_train_before_data_aug))))
                print("After data augmentation:")
                print(Counter(list(map(str, y_train_after_data_aug))))

            # Fit model.
            if config.verbose_mode:
                print("Training set size: {}".format(X_train.shape[0]))
                print("Validation set size: {}".format(X_val.shape[0]))
                print("Test set size: {}".format(X_test.shape[0]))
            model.train_model(X_train, X_val, y_train, y_val, class_weights)

        # Binary classification (binarised mini-MIAS dataset)
        elif config.dataset == "mini-MIAS-binary":
            # Import entire dataset.
            images, labels = import_minimias_dataset(data_dir="../data/{}/images_processed".format(config.dataset),
                                                     label_encoder=l_e)

            # Split dataset into training/test/validation sets (80/20% split).
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.20, dataset=images, labels=labels)

            # Create CNN model and split training/validation set (80/20% split).
            model = CnnModel(config.model, l_e.classes_.size)
            # model.load_minimias_weights()
            # model.load_minimias_fc_weights()

            # Fit model.
            if config.verbose_mode:
                print("Training set size: {}".format(X_train.shape[0]))
                print("Validation set size: {}".format(X_val.shape[0]))
            model.train_model(X_train, X_val, y_train, y_val, None)

        # Binary classification (CBIS-DDSM dataset).
        elif config.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_training_dataset(l_e)

            # Split training dataset into training/validation sets (75%/25% split).
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=images, labels=labels)
            train_dataset = create_dataset(X_train, y_train)
            validation_dataset = create_dataset(X_val, y_val)

            # Calculate class weights.
            class_weights = calculate_class_weights(y_train, l_e)

            # Create and train CNN model.
            model = CnnModel(config.model, l_e.classes_.size)
            # model.load_minimias_fc_weights()
            # model.load_minimias_weights()

            # Fit model.
            if config.verbose_mode:
                print("Training set size: {}".format(X_train.shape[0]))
                print("Validation set size: {}".format(X_val.shape[0]))
            model.train_model(train_dataset, validation_dataset, None, None, class_weights)

        # Save training runtime.
        runtime = round(time.time() - start_time, 2)

        # Save the model and its weights/biases.
        model.save_model()
        model.save_weights()

        # Evaluate training results.
        print_cli_arguments()
        if config.dataset == "mini-MIAS":
            model.make_prediction(X_val)
            model.evaluate_model(y_val, l_e, 'N-B-M', runtime)
        elif config.dataset == "mini-MIAS-binary":
            model.make_prediction(X_val)
            model.evaluate_model(y_val, l_e, 'B-M', runtime)
        elif config.dataset == "CBIS-DDSM":
            model.make_prediction(validation_dataset)
            model.evaluate_model(y_val, l_e, 'B-M', runtime)
        print_runtime("Training", runtime)

    # Run in testing mode.
    elif config.run_mode == "test":

        print("-- Testing model --\n")

        # Start recording time.
        start_time = time.time()

        # Test multi-class classification (mini-MIAS dataset).
        if config.dataset == "mini-MIAS":
            images, labels = import_minimias_dataset(data_dir="../data/{}/images_processed".format(config.dataset),
                                                     label_encoder=l_e)
            _, X_test, _, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels)
            model = load_trained_model()
            predictions = model.predict(x=X_test)
            runtime = round(time.time() - start_time, 2)
            test_model_evaluation(y_test, predictions, l_e, 'N-B-M', runtime)

        # Test binary classification (binarised mini-MIAS dataset).
        elif config.dataset == "mini-MIAS-binary":
            pass

        # Test binary classification (CBIS-DDSM dataset).
        elif config.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_testing_dataset(l_e)
            test_dataset = create_dataset(images, labels)
            model = load_trained_model()
            predictions = model.predict(x=test_dataset)
            runtime = round(time.time() - start_time, 2)
            test_model_evaluation(labels, predictions, l_e, 'B-M', runtime)

        print_runtime("Testing", runtime)


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their value in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default="CBIS-DDSM",
                        required=True,
                        help="The dataset to use. Must be either 'mini-MIAS', 'mini-MIAS-binary' or 'CBIS-DDMS'."
                        )
    parser.add_argument("-mt", "--mammogramtype",
                        default="all",
                        help="The mammogram type to use. Can be either 'calc', 'mass' or 'all'. Defaults to 'all'."
                        )
    parser.add_argument("-m", "--model",
                        required=True,
                        help="The model to use. Must be either 'VGG-common', 'VGG', 'ResNet', 'Inception', 'DenseNet', 'MobileNet' or 'CNN'."
                        )
    parser.add_argument("-r", "--runmode",
                        default="train",
                        help="The mode to run the code in. Either train the model from scratch and make predictions, "
                             "otherwise load a previously trained model for predictions. Must be either 'train' or "
                             "'test'. Defaults to 'train'."
                        )
    parser.add_argument("-lr", "--learning-rate",
                        type=float,
                        default=1e-3,
                        help="The learning rate for the non-ImageNet-pre-trained layers. Defaults to 1e-3."
                        )
    parser.add_argument("-b", "--batchsize",
                        type=int,
                        default=2,
                        help="The batch size to use. Defaults to 2."
                        )
    parser.add_argument("-e1", "--max_epoch_frozen",
                        type=int,
                        default=100,
                        help="The maximum number of epochs in the first training phrase (with frozen layers). Defaults "
                             "to 100."
                        )
    parser.add_argument("-e2", "--max_epoch_unfrozen",
                        type=int,
                        default=50,
                        help="The maximum number of epochs in the second training phrase (with unfrozen layers). "
                             "Defaults to 50."
                        )
    # parser.add_argument("-gs", "--gridsearch",
    #                    action="store_true",
    #                    default=False,
    #                    help="Include this flag to run the grid search algorithm to determine the optimal "
    #                         "hyperparameters for the CNN model."
    #                    )
    parser.add_argument("-roi", "--roi",
                        action="store_true",
                        default=False,
                        help="Include this flag to use a cropped version of the images around the ROI. Only use with 'mini-MIAS' dataset."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    parser.add_argument("-n", "--name",
                        default="",
                        help="The name of the experiment being tested. Defaults to an empty string."
                        )

    args = parser.parse_args()
    config.dataset = args.dataset
    config.mammogram_type = args.mammogramtype
    config.model = args.model
    config.run_mode = args.runmode
    if args.learning_rate <= 0:
        print_error_message()
    config.learning_rate = args.learning_rate
    if args.batchsize <= 0 or args.batchsize >= 25:
        print_error_message()
    config.batch_size = args.batchsize
    if all([args.max_epoch_frozen, args.max_epoch_unfrozen]) <= 0:
        print_error_message()
    config.max_epoch_frozen = args.max_epoch_frozen
    config.max_epoch_unfrozen = args.max_epoch_unfrozen
    # config.is_grid_search = args.gridsearch
    config.is_roi = args.roi
    config.verbose_mode = args.verbose
    config.name = args.name

    if config.verbose_mode:
        print_cli_arguments()


if __name__ == '__main__':
    main()
