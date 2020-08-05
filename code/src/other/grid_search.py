# from hyperas import optim
# from hyperas.distributions import choice, uniform
# from hyperopt import STATUS_OK, Trials, tpe
# import numpy as np
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
#
# import config
#
#
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
#
#
# def create_model(X_train, y_train):
#     base_model = Sequential(name="GS_Base_Model")
#
#     single_channel_input = Input(shape=(config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE['WIDTH'], 1))
#     triple_channel_input = Concatenate()([single_channel_input, single_channel_input, single_channel_input])
#     input_model = Model(inputs=single_channel_input, outputs=triple_channel_input)
#     base_model.add(input_model)
#
#     # Generate extra convolutional layers for model to put at the beginning
#     base_model.add(Conv2D(16, (3, 3),
#                           activation='relu',
#                           padding='same'))
#     base_model.add(Conv2D(16, (3, 3),
#                           activation='relu',
#                           padding='same'))
#     base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     base_model.add(Dropout({{uniform(0, 1)}}))
#
#     # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluding the fully
#     # connected layers.
#     base_model.add(Conv2D(64, (3, 3),
#                           activation='relu',
#                           padding='same'))
#     pre_trained_model = VGG19(include_top=False,
#                               weights="imagenet",
#                               input_shape=[config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 3])
#
#     # Exclude input layer and first convolutional layer of VGG model.
#     pre_trained_model_trimmed = Sequential(name="GS_Pre-trained_Model")
#     for layer in pre_trained_model.layers[2:]:
#         pre_trained_model_trimmed.add(layer)
#
#     # Add fully connected layers
#     model = Sequential(name="GS_Breast_Cancer_Model")
#
#     # Start with base model consisting of convolutional layers
#     model.add(base_model)
#     model.add(pre_trained_model_trimmed)
#
#     # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
#     model.add(Flatten())
#
#     # Add fully connected hidden layers and dropout layers between each for regularisation.
#     model.add(Dense(units=512, activation='relu', name='Dense_1'))
#     #         self._model.add(Dropout(0.2))
#     model.add(Dense(units=32, activation='relu', name='Dense_2'))
#     #         self._model.add(Dropout(0.2))
#     #         self._model.add(Dense(units=16, activation='relu', name='Dense_3'))
#     #         self._model.add(Dropout(0.2))
#
#     # Final output layer that uses softmax activation function (because the classes are exclusive).
#     if config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
#         model.add(Dense(1, activation='sigmoid', name='Output'))
#     elif config.dataset == "mini-MIAS":
#         model.add(Dense(3, activation='softmax', name='Output'))
#
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
#                   optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
#
#     result = model.fit(X_train, y_train,
#                        batch_size=config.batch_size,
#                        epochs=config.max_epoch_unfrozen,
#                        verbose=2,
#                        validation_split=0.25)
#
#     validation_acc = np.amax(result.history['val_acc'])
#     print('Best validation acc of epoch:', validation_acc)
#
#     return {
#         'loss': -validation_acc,
#         'status': STATUS_OK,
#         'model': model
#     }
#
#
#     def grid_search(self, X_train, y_train) -> None:
#         """
#         Perform grid search.
#         :param X_train:
#         :param y_train:
#         :return: Nones
#         """
#         param_grid = {
#             "optimizer": ["rmsprop", "adam"]
#         }
#
#         # Wrap Keras model for Sklearn grid search.
#         self.compile_model(1e-3)
#         model = KerasClassifier(build_fn=self.model)
#
#         # Grid Search
#         gs = GridSearchCV(
#             estimator=model,
#             param_grid=param_grid,
#             cv=5,
#             n_jobs=-1,
#             scoring=make_scorer(log_loss)
#         )
#         gs_results = gs.fit(X_train, y_train)
#
#         # Save results to CSV.
#         gs_results_df = pd.DataFrame(gs_results.cv_results_)
#         gs_results_df.to_csv(
#             "../output/dataset-{}_model-{}_b-{}_e1-{}_e2-{}_grid_search_results.csv".format(
#                 config.dataset,
#                 config.model,
#                 config.batch_size,
#                 config.max_epoch_frozen,
#                 config.max_epoch_unfrozen
#             ))
#
#         # Save best model found.
#         final_model = gs_results.best_estimator_
#         print("\nBest model hyperparameters found by grid search algorithm:")
#         print(final_model)
#         print("Score: {}".format(gs_results.best_score_))
#         joblib.dump(
#             final_model,
#             "/cs/scratch/agj6/saved_models/dataset-{}_model-{}_b-{}_e1-{}_e2-{}_gs-best-estimator.pkl".format(
#                 config.dataset,
#                 config.model,
#                 config.batch_size,
#                 config.max_epoch_frozen,
#                 config.max_epoch_unfrozen
#             ))
