import argparse
import time

from tensorflow.keras.models import load_model

import config
from data_operations.dataset_feed import create_dataset
from data_operations.data_preprocessing import dataset_stratified_split, generate_image_transforms, \
    import_cbisddsm_training_dataset, import_minimias_dataset
from model.cnn_model import CNN_Model
from model.vgg_model_large import generate_vgg_model_large
from utils import create_label_encoder, print_cli_arguments, print_error_message, \
    print_num_gpus_available, print_runtime


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which dataset and model to use.
    :return: None.
    """
    parse_command_line_arguments()
    print_num_gpus_available()

    # Start recording time.
    start_time = time.time()

    # Create label encoder.
    l_e = create_label_encoder()

    # Run in training mode.
    if config.run_mode == "train":

        # Multi-class classification (mini-MIAS dataset)
        if config.dataset == "mini-MIAS":
            # Import entire dataset.
            images, labels = import_minimias_dataset(data_dir="../data/{}/images_processed".format(config.dataset),
                                                     label_encoder=l_e)

            # Split dataset into training/test/validation sets (60%/20%/20% split).
            X_train, X_test, y_train, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels)
            X_train_rebalanced, y_train_rebalanced = generate_image_transforms(X_train, y_train)
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=X_train_rebalanced,
                                                                      labels=y_train_rebalanced)
            # Create and train CNN model.
            model = CNN_Model(config.model, l_e.classes_.size)
            model.train_model(X_train, y_train, X_val, y_val, config.batch_size, config.max_epoch_frozen,
                              config.max_epoch_unfrozen)

        # Binary classification (CBIS-DDSM dataset).
        elif config.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_training_dataset(l_e)

            # Split training dataset into training/validation sets (75%/25% split).
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=images, labels=labels)
            train_dataset = create_dataset(X_train, y_train)
            validation_dataset = create_dataset(X_val, y_val)

            # Create and train CNN model.

            if config.image_size == "small":
                model = CNN_Model(config.model, l_e.classes_.size)
            else:
                model = generate_vgg_model_large(l_e.classes_.size)

            model.train_model(train_dataset, None, validation_dataset, None, config.batch_size, config.max_epoch_frozen,
                              config.max_epoch_unfrozen)

        else:
            print_error_message()

        # Save the model
        model.save_model()

    elif config.run_mode == "test":
        model = load_model("../saved_models/dataset-{}_model-{}_imagesize-{}.h5".format(config.dataset, config.model,
                                                                                        config.image_size))
    # Evaluate model results.
    print_cli_arguments()
    if config.dataset == "mini-MIAS":
        model.make_prediction(X_val)
        model.evaluate_model(y_val, l_e, config.dataset, 'N-B-M')
    elif config.dataset == "CBIS-DDSM":
        model.make_prediction(validation_dataset)
        model.evaluate_model(y_val, l_e, config.dataset, 'B-M')

    # Print training runtime.
    print_runtime("Total", round(time.time() - start_time, 2))


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
                        required=True,
                        help="The model to use. Must be either 'VGG' or 'Inception'."
                        )
    parser.add_argument("-r", "--runmode",
                        default="train",
                        help="The mode to run the code in. Either train the model from scratch and make predictions, "
                             "otherwise load a previously trained model for predictions. Must be either 'train' or "
                             "'test'. Defaults to 'train'."
                        )
    parser.add_argument("-i", "--imagesize",
                        default="small",
                        help="The initial input image size to feed into the CNN model. If set to 'small', will use "
                             "images resized to 512x512px. If set to 'large' will use images resized to 2048x2048px "
                             "(using with extra convolution layers for downsizing). Defaults to 'small'."
                        )
    parser.add_argument("-b", "--batchsize",
                        type=int,
                        default=2,
                        help="The batch size to use Defaults to 'small'."
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
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )

    args = parser.parse_args()
    config.dataset = args.dataset
    config.model = args.model
    config.run_mode = args.runmode
    config.image_size = args.imagesize
    if args.batchsize <= 0 or args.batchsize >= 25:
        print_error_message()
    config.batch_size = args.batchsize
    if all([args.max_epoch_frozen, args.max_epoch_unfrozen]) <= 0:
        print_error_message()
    config.max_epoch_frozen = args.max_epoch_frozen
    config.max_epoch_unfrozen = args.max_epoch_unfrozen
    config.verbose_mode = args.verbose
    
    if config.verbose_mode:
        print_cli_arguments()


if __name__ == '__main__':
    main()
