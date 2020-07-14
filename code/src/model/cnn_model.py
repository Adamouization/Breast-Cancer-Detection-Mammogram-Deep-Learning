import json
import ssl

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import InceptionV3, ResNet50, ResNet50V2, VGG19, Xception
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

from data_visualisation.plots import *
from data_visualisation.roc_curves import *

# Required to download pre-trained weights for ImageNet (stored in ~/.keras/models/)
ssl._create_default_https_context = ssl._create_unverified_context


class CNN_Model:

    def __init__(self, model_name: str, num_classes: int):
        """
        Function to create a VGG19 model pre-trained with custom FC Layers.
        If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to
        support larger images.
        :param model_name: The CNN model to use.
        :param num_classes: The number of classes (labels).
        :return: The VGG19 model.
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.prediction = None

        self.create_model()

    def create_model(self):
        """
        Creates a CNN from an existing architecture with pre-trained weights on ImageNet.
        """
        # Reconfigure a single channel image input (greyscale) into a 3-channel greyscale input (tensor).
        if self.model_name == "VGG":
            single_channel_input = Input(shape=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 1))
        elif self.model_name == "ResNet":
            single_channel_input = Input(shape=(config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE['WIDTH'], 1))
        elif self.model_name == "Inception":
            single_channel_input = Input(
                shape=(config.INCEPTION_IMG_SIZE['HEIGHT'], config.INCEPTION_IMG_SIZE['WIDTH'], 1))
        elif self.model_name == "Xception":
            single_channel_input = Input(
                shape=(config.XCEPTION_IMG_SIZE['HEIGHT'], config.XCEPTION_IMG_SIZE['WIDTH'], 1)
            )
        triple_channel_input = Concatenate()([single_channel_input, single_channel_input, single_channel_input])

        # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluding the fully
        # connected layers.
        if self.model_name == "VGG":
            base_model = VGG19(include_top=False, weights="imagenet", input_tensor=triple_channel_input)
        elif self.model_name == "ResNet":
            base_model = ResNet50V2(include_top=False, weights="imagenet", input_tensor=triple_channel_input)
        elif self.model_name == "Inception":
            base_model = InceptionV3(include_top=False, weights="imagenet", input_tensor=triple_channel_input)
        elif self.model_name == "Xception":
            base_model = Xception(include_top=False, weights="imagenet", input_tensor=triple_channel_input)

        # Add fully connected layers
        self._model = Sequential()

        # Start with base model consisting of convolutional layers
        self._model.add(base_model)

        # Generate additional convolutional layers (advanced model)
        # self._model.add(Conv2D(1024, (3, 3),
        #                        activation='relu',
        #                        padding='same'))
        # self._model.add(Conv2D(1024, (3, 3),
        #                        activation='relu',
        #                        padding='same'))
        # self._model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
        self._model.add(Flatten())

        # Add fully connected hidden layers.
        self._model.add(Dense(units=512, activation='relu', name='Dense_1'))
        self._model.add(Dense(units=128, activation='relu', name='Dense_2'))
        self._model.add(Dense(units=32, activation='relu', name='Dense_3'))

        # Possible dropout for regularisation can be added later and experimented with:
        # model.add(Dropout(0.1, name='Dropout_Regularization'))

        # Final output layer that uses softmax activation function (because the classes are exclusive).
        if self.num_classes == 2:
            self._model.add(Dense(1, activation='sigmoid', name='Output'))
        else:
            self._model.add(Dense(self.num_classes, activation='softmax', name='Output'))

        # Print model details if running in debug mode.
        if config.verbose_mode:
            print(self._model.summary())
            
        

    def train_model(self, train_x, train_y, val_x, val_y, batch_s, epochs1, epochs2):
        """
        Function to train network in two steps:
        * Train network with initial VGG base layers frozen
        * Unfreeze all layers and retrain with smaller learning rate
        :param model: CNN model
        :param train_x: training input
        :param train_y: training outputs
        :param val_x: validation inputs
        :param val_y: validation outputs
        :param batch_s: batch size
        :param epochs1: epoch count for initial training
        :param epochs2: epoch count for training all layers unfrozen
        :return: trained network
        """
        # Freeze pre-trained CNN model layers: only train fully connected layers.
        if config.image_size == "large":
            self._model.layers[0].layers[1].trainable = False
        else:
            self._model.layers[0].trainable = False

        # Train model with frozen layers (all training with early stopping dictated by loss in validation over 3 runs).

        if config.dataset == "mini-MIAS":
            self._model.compile(optimizer=Adam(1e-3),
                                loss=CategoricalCrossentropy(),
                                metrics=[CategoricalAccuracy()])
            hist_1 = self._model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs1,
                callbacks=[
                    EarlyStopping(monitor='val_categorical_accuracy', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)
                ]
            )
        
        elif config.dataset == "mini-MIAS-binary":
            self._model.compile(optimizer=Adam(1e-3),
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
            hist_1 = self._model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs1,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)
                ]
            )

        elif config.dataset == "CBIS-DDSM":
            self._model.compile(optimizer=Adam(lr=1e-3),
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
            hist_1 = self._model.fit(
                x=train_x,
                validation_data=val_x,
                epochs=epochs1,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)]
                )

        # Plot the training loss and accuracy.
        plot_training_results(hist_1, "Initial_training", True)

        # Train a second time with a smaller learning rate and with all layers unfrozen
        # (train over fewer epochs to prevent over-fitting).
        if config.image_size == "large":
            self._model.layers[0].layers[1].trainable = True
        else:
            self._model.layers[0].trainable = True

        if config.dataset == "mini-MIAS":
            self._model.compile(optimizer=Adam(1e-5),  # Very low learning rate
                                loss=CategoricalCrossentropy(),
                                metrics=[CategoricalAccuracy()])
            hist_2 = self._model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs2,
                callbacks=[
                    EarlyStopping(monitor='val_categorical_accuracy', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=6)
                ]
            )
        
        elif config.dataset == "mini-MIAS-binary":
            self._model.compile(optimizer=Adam(1e-5),  # Very low learning rate
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
            hist_2 = self._model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs2,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=6)
                ]
            )
        
        elif config.dataset == "CBIS-DDSM":
            self._model.compile(optimizer=Adam(lr=1e-5),  # Very low learning rate
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
            hist_2 = self._model.fit(
                x=train_x,
                validation_data=val_x,
                epochs=epochs2,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=6)]
                )

        # Plot the training loss and accuracy.
        plot_training_results(hist_2, "Fine_tuning_training", False)

    def grid_search(self, X_train, y_train) -> None:
        """Grid Search"""
        print("doing grid search")
        param_grid = {
            "optimizer": ["rmsprop", "adam"]
        }
        
        self._model.compile(optimizer=Adam(1e-3),
                            loss=CategoricalCrossentropy(),
                            metrics=[CategoricalAccuracy()])
        
        # Wrap Keras model for Sklearn grid search.
        model = KerasClassifier(build_fn=self.model)
        
        # Grid Search
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid, 
            cv=5, 
            n_jobs=-1, 
            scoring=make_scorer(log_loss)
        )
        gs_results = gs.fit(X_train, y_train)
        
        # Save results to CSV.
        gs_results_df = pd.DataFrame(gs_results.cv_results_)
        gs_results_df.to_csv("../output/dataset-{}_model-{}_imagesize-{}_b-{}_e1-{}_e2-{}_grid_search_results.csv".format(
            config.dataset,
            config.model,
            config.image_size,
            config.batch_size,
            config.max_epoch_frozen,
            config.max_epoch_unfrozen
        ))
        
        # Save best model found.
        final_model = gs_results.best_estimator_
        print("\nBest model hyperparameters found by grid search algorithm:")
        print(final_model)
        print("Score: {}".format(gs_results.best_score_))
        joblib.dump(final_model, "/cs/scratch/agj6/saved_models/dataset-{}_model-{}_imagesize-{}_b-{}_e1-{}_e2-{}_gs-best-estimator.pkl".format(
            config.dataset,
            config.model,
            config.image_size,
            config.batch_size,
            config.max_epoch_frozen,
            config.max_epoch_unfrozen
        ))
        
        
        
    def evaluate_model(self, y_true: list, label_encoder: LabelEncoder, classification_type: str):
        """
        Evaluate model performance with accuracy, confusion matrix, ROC curve and compare with other papers' results.
        :param y_true: Ground truth of the data in one-hot-encoding type.
        :param y_pred: Prediction result of the data in one-hot-encoding type.
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
            "../output/dataset-{}_model-{}_imagesize-{}_b-{}_e1-{}_e2-{}_report.csv".format(config.dataset,
                                                                                            config.model,
                                                                                            config.image_size,
                                                                                            config.batch_size,
                                                                                            config.max_epoch_frozen,
                                                                                            config.max_epoch_unfrozen),
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
        with open('data_visualisation/other_paper_results.json') as config_file:  # Load other papers' results from JSON.
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

    def make_prediction(self, x_values):
        """
        :param x: Input.
        :return: Model predictions.
        """
        if config.dataset == "mini-MIAS" or "mini-MIAS-binary":
            self.prediction = self._model.predict(x=x_values.astype("float32"), batch_size=config.batch_size)
        elif config.dataset == "CBIS-DDSM":
            self.prediction = self._model.predict(x=x_values)
        # if config.verbose_mode:
        #   print("Predictions:")
        #   print(self.prediction)

    def save_model(self):
        # Scratch space
        self._model.save(
            "/cs/scratch/agj6/saved_models/dataset-{}_model-{}_imagesize-{}_b-{}_e1-{}_e2-{}.h5".format(
                config.dataset,
                config.model,
                config.image_size,
                config.batch_size,
                config.max_epoch_frozen,
                config.max_epoch_unfrozen)
        )
        # Local directory below
        # self._model.save("../saved_models/dataset-{}_model-{}_imagesize-{}_b-{}_e1-{}_e2-{}.h5".format(config.dataset, config.model, config.image_size, config.batch_size, config.max_epoch_frozen, config.max_epoch_unfrozen))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        pass
