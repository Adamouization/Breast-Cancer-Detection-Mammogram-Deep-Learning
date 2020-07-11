import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


def plot_confusion_matrix(cm: np.ndarray, fmt: str, label_encoder, is_normalised: bool) -> None:
    """
    Plot confusion matrix.
    :param cm: Confusion matrix array.
    :param fmt: The formatter for numbers in confusion matrix.
    :param label_encoder: The label encoder used to get the number of classes.
    :param is_normalised: Boolean specifying whether the confusion matrix is normalised or not.
    :return: None.
    """
    title = str()
    if is_normalised:
        title = "Confusion Matrix Normalised"
    elif not is_normalised:
        title = "Confusion Matrix"

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, ax=ax, fmt=fmt, cmap=plt.cm.Blues)  # annot=True to annotate cells

    # Set labels, title, ticks and axis range.
    ax.set_xlabel('Predicted classes')
    ax.set_ylabel('True classes')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(label_encoder.classes_)
    ax.yaxis.set_ticklabels(label_encoder.classes_)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    bottom, top = ax.get_ylim()
    if is_normalised:
        plt.savefig("../output/dataset-{}_model-{}_imagesize-{}_CM-norm.png".format(config.dataset, config.model,
                                                                                    config.image_size))
    elif not is_normalised:
        plt.savefig(
            "../output/dataset-{}_model-{}_imagesize-{}_CM.png".format(config.dataset, config.model, config.image_size))
    plt.show()


def plot_comparison_chart(df: pd.DataFrame) -> None:
    """
    Plot comparison bar chart.
    :param df: Compare data from json file.
    :return: None.
    """
    title = "Accuracy Comparison"

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x='paper', y='accuracy', data=df)

    # Add number at the top of the bar.
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01, height, ha='center')

    # Set title.
    plt.title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.savefig(
        "../output/dataset-{}_model-{}_imagesize-{}_{}.png".format(config.dataset, config.model, config.image_size,
                                                                   title), bbox_inches='tight')
    plt.show()


def plot_training_results(hist_input, plot_name: str, is_frozen_layers) -> None:
    """
    Function to plot loss and accuracy over epoch count for training
    :param is_frozen_layers: Boolean controlling whether some layers are frozen (for the plot title).
    :param hist_input: The training history.
    :param plot_name: The plot name.
    """
    title = "Training Loss and Accuracy on Dataset"
    if not is_frozen_layers:
        title += " (all layers unfrozen)"

    n = len(hist_input.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n), hist_input.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n), hist_input.history["val_loss"], label="val_loss")
    if config.dataset == "mini-MIAS":
        plt.plot(np.arange(0, n), hist_input.history["categorical_accuracy"], label="train_acc")
        plt.plot(np.arange(0, n), hist_input.history["val_categorical_accuracy"], label="val_acc")
    elif config.dataset == "CBIS-DDSM":
        plt.plot(np.arange(0, n), hist_input.history["binary_accuracy"], label="train_acc")
        plt.plot(np.arange(0, n), hist_input.history["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(
        "../output/dataset-{}_model-{}_imagesize-{}_{}.png".format(config.dataset, config.model, config.image_size,
                                                                   plot_name))
    plt.show()
