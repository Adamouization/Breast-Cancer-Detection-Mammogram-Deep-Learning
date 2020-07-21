import json
import ssl

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, make_scorer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
from cnn_models.vgg19 import create_vgg19_model
from data_visualisation.plots import *
from data_visualisation.roc_curves import *


class CNN_Model:

    def __init__(self, model_name: str, num_classes: int):
        """
        Function to create a CNN model containing a pre-trained CNN architecture with custom convolution layers at the
        top and fully connected layers at the end.
        :param model_name: The CNN model to use.
        :param num_classes: The number of classes (labels).
        :return: The VGG19 model.
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.history = None
        self.prediction = None

        if self.model_name == "VGG":
            self._model = create_vgg19_model(self.num_classes)
        elif self.model_name == "ResNet":
            pass
        elif self.model_name == "Inception":
            pass
        elif self.model_name == "Xception":
            pass

    def train_model(self, X_train, X_val, y_train, y_val) -> None:
        """
        Function to train network in two steps:
            * Train network with initial pre-trained CNN's layers frozen.
            * Unfreeze all layers and retrain with smaller learning rate.
        Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
        :param X_train: training input
        :param X_val: training outputs
        :param y_train: validation inputs
        :param y_val: validation outputs
        :return: None
        """
        # Freeze pre-trained CNN model layers: only train fully connected layers.
        self._model.layers[1].trainable = False

        # Train model with frozen layers (all training with early stopping dictated by loss in validation over 3 runs).
        self.compile_model(3e-3)
        self.fit_model(X_train, X_val, y_train, y_val, )
        # Plot the training loss and accuracy.
        plot_training_results(self.history, "Initial_training", True)

        # Unfreeze all layers.
        self._model.layers[1].trainable = True

        # Train a second time with a smaller learning rate (train over fewer epochs to prevent over-fitting).
        self.compile_model(1e-5)  # Very low learning rate.
        self.fit_model(X_train, X_val, y_train, y_val)
        # Plot the training loss and accuracy.
        plot_training_results(self.history, "Fine_tuning_training", False)

    def compile_model(self, learning_rate) -> None:
        """
        Compile the Keras CNN model.
        Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
        :param learning_rate: The initial learning rate for the optimiser.
        :return: None
        """
        if config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
            self._model.compile(optimizer=Adam(learning_rate),
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
        elif config.dataset == "mini-MIAS":
            self._model.compile(optimizer=Adam(learning_rate),
                                loss=CategoricalCrossentropy(),
                                metrics=[CategoricalAccuracy()])

    def fit_model(self, X_train, X_val, y_train, y_val, is_frozen_layers: bool) -> None:
        """
        Fit the Keras CNN model and plot the training evolution.
        Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
        :param X_train:
        :param X_val:
        :param y_train:
        :param y_val:
        :param is_frozen_layers:
        :return:
        """
        if is_frozen_layers:
            max_epochs = config.max_epoch_frozen
        else:
            max_epochs = config.max_epoch_unfrozen

        if config.dataset == "mini-MIAS":
            self.history = self._model.fit(
                x=X_train,
                y=y_train,
                batch_size=config.batch_size,
                steps_per_epoch=len(X_train) // config.batch_size,
                validation_data=(X_val, y_val),
                validation_steps=len(X_val) // config.batch_size,
                epochs=max_epochs,
                callbacks=[
                    EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)
                ]
            )
        elif config.dataset == "mini-MIAS-binary":
            self.history = self._model.fit(
                x=X_train,
                y=y_train,
                batch_size=config.batch_size,
                steps_per_epoch=len(X_train) // config.batch_size,
                validation_data=(X_val, y_val),
                validation_steps=len(X_val) // config.batch_size,
                epochs=max_epochs,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)
                ]
            )
        elif config.dataset == "CBIS-DDSM":
            self.history = self._model.fit(
                x=X_train,
                validation_data=X_val,
                epochs=config.max_epochs,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)]
            )

    def make_prediction(self, x_values):
        """
        Makes a prediction using unseen data.
        Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
        :param x_values: The input.
        :return: The model predictions (not probabilities).
        """
        if config.dataset == "mini-MIAS" or config.dataset == "mini-MIAS-binary":
            self.prediction = self._model.predict(x=x_values.astype("float32"), batch_size=config.batch_size)
        elif config.dataset == "CBIS-DDSM":
            self.prediction = self._model.predict(x=x_values)
        # print(self.prediction)

    def evaluate_model(self, y_true: list, label_encoder: LabelEncoder, classification_type: str) -> None:
        """
        Evaluate model performance with accuracy, confusion matrix, ROC curve and compare with other papers' results.
        Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
        :param y_true: Ground truth of the data in one-hot-encoding type.
        :param label_encoder: The label encoder for y value (label).
        :param classification_type: The classification type. Ex: N-B-M: normal, benign and malignant; B-M: benign and
        malignant.
        :return: None.
        """
        # Inverse transform y_true and y_pred from one-hot-encoding to original label.
        if label_encoder.classes_.size == 2:
            y_true_inv = y_true
            y_pred_inv = np.round_(self.prediction, 0)
        else:
            y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
            y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))

        # Calculate accuracy.
        accuracy = float('{:.4f}'.format(accuracy_score(y_true_inv, y_pred_inv)))
        print("Accuracy = {}\n".format(accuracy))

        # Print and save classification report for precision, recall and f1 score metrics.
        print(classification_report(y_true_inv, y_pred_inv, target_names=label_encoder.classes_))
        report_df = pd.DataFrame(classification_report(y_true_inv, y_pred_inv, target_names=label_encoder.classes_,
                                                       output_dict=True)).transpose()
        report_df.append({'accuracy': accuracy}, ignore_index=True)
        report_df.to_csv(
            "../output/dataset-{}_model-{}_b-{}_e1-{}_e2-{}_report.csv".format(
                config.dataset,
                config.model,
                config.batch_size,
                config.max_epoch_frozen,
                config.max_epoch_unfrozen
            ),
            index=False,
            header=True
        )

        # Plot confusion matrix and normalised confusion matrix.
        cm = confusion_matrix(y_true_inv, y_pred_inv)  # Calculate CM with original label of classes
        plot_confusion_matrix(cm, 'd', label_encoder, False)
        # Calculate normalized confusion matrix with original label of classes.
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized[np.isnan(cm_normalized)] = 0
        plot_confusion_matrix(cm_normalized, '.2f', label_encoder, True)

        # Plot ROC curve.
        if label_encoder.classes_.size == 2:  # binary classification
            plot_roc_curve_binary(y_true, self.prediction)
        elif label_encoder.classes_.size >= 2:  # multi classification
            plot_roc_curve_multiclass(y_true, self.prediction, label_encoder)

        # Compare results with other similar papers' result.
        with open(
                '../data_visualisation/other_paper_results.json') as config_file:  # Load other papers' results from JSON.
            data = json.load(config_file)

        dataset_key = config.dataset
        if config.dataset == "mini-MIAS-binary":
            dataset_key = "mini-MIAS"

        df = pd.DataFrame.from_records(data[dataset_key][classification_type],
                                       columns=['paper', 'accuracy'])  # Filter data by dataset and classification type.
        new_row = pd.DataFrame({'paper': 'Dissertation', 'accuracy': accuracy},
                               index=[0])  # Add model result into dataframe to compare.
        df = pd.concat([new_row, df]).reset_index(drop=True)
        df['accuracy'] = pd.to_numeric(df['accuracy'])  # Digitize the accuracy column.
        plot_comparison_chart(df)

    def save_model(self) -> None:
        """
        Saves the model in h5 format.
        Currently saves in lab machines scratch space.
        :return: None
        """
        # Scratch space
        self._model.save(
            "/cs/scratch/agj6/saved_models/dataset-{}_model-{}_b-{}_e1-{}_e2-{}.h5".format(
                config.dataset,
                config.model,
                config.batch_size,
                config.max_epoch_frozen,
                config.max_epoch_unfrozen)
        )

    # def save_fully_connected_layers_weights(self):
    #     """
    #     Save the weights of the fully connected layers.
    #     :return:
    #     """
    #     weights_and_biases = self._model.layers[2].get_weights()
    #     np.save(
    #         "/cs/scratch/agj6/saved_models/dataset-{}_model-{}_b-{}_e1-{}_e2-{}.h5".format(
    #             config.dataset,
    #             config.model,
    #             config.batch_size,
    #             config.max_epoch_frozen,
    #             config.max_epoch_unfrozen),
    #         weights_and_biases)

    @property
    def model(self):
        """
        CNN model getter.
        :return: the model.
        """
        return self._model

    @model.setter
    def model(self, value) -> None:
        """
        CNN model setter.
        :param value:
        :return: None
        """
        pass