# from hyperas import optim
# from hyperas.distributions import choice, uniform
# from hyperopt import STATUS_OK, Trials, tpe
# import joblib
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, make_scorer
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.applications import InceptionV3, ResNet50, ResNet50V2, VGG19, Xception
# from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
# from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
# from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
# from tensorflow.python.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

# import config


# def fine_tune_hyperparameters(X_train, X_val, y_train, y_val):
#     best_run, best_model = optim.minimize(model=create_model,
#                                           data=data,
#                                           algo=tpe.suggest,
#                                           max_evals=5,
#                                           trials=Trials())
#     print("Evalutation of best performing model:")
#     print(best_model.evaluate(X_val, y_val))
#     print("Best performing model chosen hyper-parameters:")
#     print(best_run)

# def create_model(X_train, y_train):
#     base_model = Sequential(name="GS_Base_Model")

#     single_channel_input = Input(shape=(config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE['WIDTH'], 1))
#     triple_channel_input = Concatenate()([single_channel_input, single_channel_input, single_channel_input])
#     input_model = Model(inputs=single_channel_input, outputs=triple_channel_input)
#     base_model.add(input_model)

#     # Generate extra convolutional layers for model to put at the beginning
#     base_model.add(Conv2D(16, (3, 3),
#                           activation='relu',
#                           padding='same'))
#     base_model.add(Conv2D(16, (3, 3),
#                           activation='relu',
#                           padding='same'))
#     base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     base_model.add(Dropout({{uniform(0,1)}}))

#     # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluding the fully
#     # connected layers.
#     base_model.add(Conv2D(64, (3, 3),
#                           activation='relu',
#                           padding='same'))
#     pre_trained_model = VGG19(include_top=False, 
#                               weights="imagenet", 
#                               input_shape=[config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 3])

#     # Exclude input layer and first convolutional layer of VGG model.
#     pre_trained_model_trimmed = Sequential(name="GS_Pre-trained_Model")
#     for layer in pre_trained_model.layers[2:]:
#         pre_trained_model_trimmed.add(layer)

#     # Add fully connected layers
#     model = Sequential(name="GS_Breast_Cancer_Model")

#     # Start with base model consisting of convolutional layers
#     model.add(base_model)
#     model.add(pre_trained_model_trimmed)

#     # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
#     model.add(Flatten())

#     # Add fully connected hidden layers and dropout layers between each for regularisation.
#     model.add(Dense(units=512, activation='relu', name='Dense_1'))
#     #         self._model.add(Dropout(0.2))
#     model.add(Dense(units=32, activation='relu', name='Dense_2'))
#     #         self._model.add(Dropout(0.2))
#     #         self._model.add(Dense(units=16, activation='relu', name='Dense_3'))
#     #         self._model.add(Dropout(0.2))

#     # Final output layer that uses softmax activation function (because the classes are exclusive).
#     if config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
#         model.add(Dense(1, activation='sigmoid', name='Output'))
#     elif config.dataset == "mini-MIAS":
#         model.add(Dense(self.num_classes, activation='softmax', name='Output'))

#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

#     result = model.fit(X_train, y_train,
#                        batch_size=config.batch_size,
#                        epochs=config.epochs,
#                        verbose=2,
#                        validation_split=0.25)

#     validation_acc = np.amax(result.history['val_acc']) 
#     print('Best validation acc of epoch:', validation_acc)

#     return {
#         'loss': -validation_acc, 
#         'status': STATUS_OK, 
#         'model': model
#     }
