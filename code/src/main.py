import socket

import tensorflow as tf
# import torch


def main() -> None:
    print("Running sanity checks for GPU through PyTorch and Tensorflow")
    if socket.gethostname() == "pc5-026-l":
        test_gpu()
    else:
        print("Running sanity checks for PyTorch & Tensorflow")
        basic_import_test()


def basic_import_test() -> None:
    # PyTorch test
    # x = torch.rand(5, 3)
    # print(x)

    # TensorFlow test
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


def test_gpu() -> None:
    # PyTorch
    # print("torch.cuda.current_device() ", torch.cuda.current_device())
    # print("torch.cuda.device(0) ", torch.cuda.device(0))
    # print("torch.cuda.device_count() ", torch.cuda.device_count())
    # print("torch.cuda.get_device_name(0) ", torch.cuda.get_device_name(0))
    # print("torch.cuda.is_available() ", torch.cuda.is_available())

    # Tensorflow
    print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
    print("-----------------------")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == '__main__':
    main()
