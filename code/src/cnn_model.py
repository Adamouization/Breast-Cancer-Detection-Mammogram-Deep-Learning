import json
import ssl

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import InceptionV3, ResNet50, ResNet50V2, VGG19, Xception
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import config
from data_visualisation.plots import *
from data_visualisation.roc_curves import *

# Required to download pre-trained weights for ImageNet (stored in ~/.keras/models/)
ssl._create_default_https_context = ssl._create_unverified_context


class CNN_Model:

    def __init__(self, model_name: str, num_classes: int):
        """
        Function to create a CNN model containing a pre-trained CNN architecture with custom convolution layers at the top and fully connected layers at the end.
        :param model_name: The CNN model to use.
        :param num_classes: The number of classes (labels).
        :return: The VGG19 model.
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.history = None
        self.prediction = None

        self.create_model()

    def create_model(self) -> None:
        """
        Creates a CNN from an existing architecture with pre-trained weights on ImageNet.
        Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
        :return: None.
        """
        base_model = Sequential(name="Base_Model")

        # Reconfigure a single channel image input (greyscale) into a 3-channel greyscale input (tensor).
        single_channel_input = Input(shape=(config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE['WIDTH'], 1))
        #         if self.model_name == "VGG":
        #             single_channel_input = Input(shape=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 1))
        #         elif self.model_name == "ResNet":
        #             single_channel_input = Input(shape=(config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE['WIDTH'], 1))
        #         elif self.model_name == "Inception":
        #             single_channel_input = Input(
        #                 shape=(config.INCEPTION_IMG_SIZE['HEIGHT'], config.INCEPTION_IMG_SIZE['WIDTH'], 1))
        #         elif self.model_name == "Xception":
        #             single_channel_input = Input(
        #                 shape=(config.XCEPTION_IMG_SIZE['HEIGHT'], config.XCEPTION_IMG_SIZE['WIDTH'], 1)
        #             )
        triple_channel_input = Concatenate()([single_channel_input, single_channel_input, single_channel_input])
        input_model = Model(inputs=single_channel_input, outputs=triple_channel_input)
        base_model.add(input_model)

        # Generate extra convolutional layers for model to put at the beginning
        base_model.add(Conv2D(16, (3, 3),
                              activation='relu',
                              padding='same'))
        base_model.add(Conv2D(16, (3, 3),
                              activation='relu',
                              padding='same'))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        base_model.add(Dropout(0.25))

        # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluding the fully
        # connected layers.
        if self.model_name == "VGG":
            base_model.add(Conv2D(64, (3, 3),
                                  activation='relu',
                                  padding='same'))
            pre_trained_model = VGG19(include_top=False, weights="imagenet",
                                      input_shape=[config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 3])
        elif self.model_name == "ResNet":
            pre_trained_model = ResNet50V2(include_top=False, weights="imagenet",
                                           input_shape=[config.RESNET_IMG_SIZE['HEIGHT'],
                                                        config.RESNET_IMG_SIZE['WIDTH'], 3])
        elif self.model_name == "Inception":
            pre_trained_model = InceptionV3(include_top=False, weights="imagenet",
                                            input_shape=[config.INCEPTION_IMG_SIZE['HEIGHT'],
                                                         config.INCEPTION_IMG_SIZE['WIDTH'], 3])
        elif self.model_name == "Xception":
            pre_trained_model = Xception(include_top=False, weights="imagenet",
                                         input_shape=[config.XCEPTION_IMG_SIZE['HEIGHT'],
                                                      config.XCEPTION_IMG_SIZE['WIDTH'], 3])

        # Exclude input layer and first convolutional layer of VGG model.
        pre_trained_model_trimmed = Sequential(name="Pre-trained_Model")
        for layer in pre_trained_model.layers[2:]:
            pre_trained_model_trimmed.add(layer)

        # Add fully connected layers
        self._model = Sequential(name="Breast_Cancer_Model")

        # Start with base model consisting of convolutional layers
        self._model.add(base_model)
        self._model.add(pre_trained_model_trimmed)

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

        # Add fully connected hidden layers and dropout layers between each for regularisation.
        self._model.add(Dense(units=512, activation='relu', name='Dense_1'))
        #         self._model.add(Dropout(0.2))
        self._model.add(Dense(units=32, activation='relu', name='Dense_2'))
        #         self._model.add(Dropout(0.2))
        #         self._model.add(Dense(units=16, activation='relu', name='Dense_3'))
        #         self._model.add(Dropout(0.2))

        # Final output layer that uses softmax activation function (because the classes are exclusive).
        if config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
            self._model.add(Dense(1, activation='sigmoid', name='Output'))
        elif config.dataset == "mini-MIAS":
            self._model.add(Dense(self.num_classes, activation='softmax', name='Output'))

        # Print model details if running in debug mode.
        if config.verbose_mode:
            print(base_model.summary())
            print(pre_trained_model_trimmed.summary())
            print(self._model.summary())

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
        self.compile_model(1e-3)
        self.fit_model(X_train, X_val, y_train, y_val)
        # Plot the training loss and accuracy.
        plot_training_results(self.history, "Initial_training", True)

        # Unfreeze all layers.
        self._model.layers[1].trainable = True

        # Train a second time with a smaller learning rate (train over fewer epochs to prevent over-fitting).
        self.compile_model(1e-4)  # Very low learning rate.
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

    def fit_model(self, X_train, X_val, y_train, y_val) -> None:
        """
        Fit the Keras CNN model and plot the training evolution.
        Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
        :param X_train:
        :param X_val:
        :param y_train:
        :param y_val:
        :return:
        """
        if config.dataset == "mini-MIAS":
            self.history = self._model.fit(
                x=X_train,
                y=y_train,
                batch_size=config.batch_size,
                steps_per_epoch=len(X_train) // config.batch_size,
                validation_data=(X_val, y_val),
                validation_steps=len(X_val) // config.batch_size,
                epochs=config.max_epoch_frozen,
                callbacks=[
                    EarlyStopping(monitor='val_categorical_accuracy', patience=8, restore_best_weights=True),
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
                epochs=config.max_epoch_frozen,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)
                ]
            )
        elif config.dataset == "CBIS-DDSM":
            self.history = self._model.fit(
                x=X_train,
                validation_data=X_val,
                epochs=config.max_epoch_frozen,
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

    def grid_search(self, X_train, y_train) -> None:
        """
        Perform grid search.
        :param X_train:
        :param y_train:
        :return: Nones
        """
        param_grid = {
            "optimizer": ["rmsprop", "adam"]
        }

        # Wrap Keras model for Sklearn grid search.
        self.compile_model(1e-3)
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
        gs_results_df.to_csv(
            "../output/dataset-{}_model-{}_b-{}_e1-{}_e2-{}_grid_search_results.csv".format(
                config.dataset,
                config.model,
                config.batch_size,
                config.max_epoch_frozen,
                config.max_epoch_unfrozen
            ))

        # Save best model found.
        final_model = gs_results.best_estimator_
        print("\nBest model hyperparameters found by grid search algorithm:")
        print(final_model)
        print("Score: {}".format(gs_results.best_score_))
        joblib.dump(
            final_model,
            "/cs/scratch/agj6/saved_models/dataset-{}_model-{}_b-{}_e1-{}_e2-{}_gs-best-estimator.pkl".format(
                config.dataset,
                config.model,
                config.batch_size,
                config.max_epoch_frozen,
                config.max_epoch_unfrozen
            ))

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
                'data_visualisation/other_paper_results.json') as config_file:  # Load other papers' results from JSON.
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
