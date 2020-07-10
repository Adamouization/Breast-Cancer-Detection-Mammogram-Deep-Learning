import argparse
import time

import config
from data_operations.dataset_feed import create_dataset
from data_operations.data_preprocessing import import_cbisddsm_training_dataset, import_minimias_dataset, \
    dataset_stratified_split, generate_image_transforms
from data_visualisation.output import evaluate
from model.train_test_model import make_predictions, train_network
from model.vgg_model import generate_vgg_model
from model.vgg_model_large import generate_vgg_model_large
from utils import create_label_encoder, print_error_message, print_num_gpus_available, print_runtime
from tensorflow.keras.models import load_model

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

        # Multiclass classification (mini-MIAS dataset)
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
            model = generate_vgg_model(l_e.classes_.size)
            model = train_network(model, X_train, y_train, X_val, y_val, config.BATCH_SIZE, config.EPOCH_1,
                                  config.EPOCH_2)

        # Binary classification (CBIS-DDSM dataset).
        elif config.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_training_dataset(l_e)

            # Split training dataset into training/validation sets (75%/25% split).
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=images, labels=labels)
            dataset_train = create_dataset(X_train, y_train)
            dataset_val = create_dataset(X_val, y_val)

            # Create and train CNN model.

            if config.imagesize == "small":
                model = generate_vgg_model(l_e.classes_.size)
            else:
                model = generate_vgg_model_large(l_e.classes_.size)

            model = train_network(model, dataset_train, None, dataset_val, None, config.BATCH_SIZE, config.EPOCH_1,
                                  config.EPOCH_2)

        else:
            print_error_message()

        # Save the model
        model.save("../saved_models/dataset-{}_model-{}_imagesize-{}.h5".format(config.dataset, config.model, config.imagesize))

    elif config.run_mode == "test":
        model = load_model("../saved_models/dataset-{}_model-{}_imagesize-{}.h5".format(config.dataset, config.model, config.imagesize))

    # Evaluate model results.
    if config.dataset == "mini-MIAS":
        y_pred = make_predictions(model, X_val)
        evaluate(y_val, y_pred, l_e, config.dataset, 'N-B-M')
    elif config.dataset == "CBIS-DDSM":
        y_pred = make_predictions(model, dataset_val)
        evaluate(y_val, y_pred, l_e, config.dataset, 'B-M')

    # Print the prediction
#     print(y_pred)

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
                        default="basic",
                        required=True,
                        help="The model to use. Must be either 'basic' or 'advanced'."
                        )
    parser.add_argument("-r", "--runmode",
                        default="train",
                        help="Running mode: train model from scratch and make predictions, otherwise load pre-trained "
                             "model for predictions. Must be either 'train' or 'test'."
                        )
    parser.add_argument("-i", "--imagesize",
                        default="small",
                        help="small: use resized images to 512x512, otherwise use 'large' to use 2048x2048 size image with model with extra convolutions for downsizing."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )

    args = parser.parse_args()
    config.dataset = args.dataset
    config.model = args.model
    config.run_mode = args.runmode
    config.imagesize = args.imagesize
    config.verbose_mode = args.verbose
    

if __name__ == '__main__':
    main()
