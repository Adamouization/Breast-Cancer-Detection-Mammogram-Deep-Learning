import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder

import config


def plot_roc_curve_binary(y_true: list, y_pred: list) -> None:
    """
    Plot ROC curve for binary classification.
    :param y_true: Ground truth of the data in one-hot-encoding type.
    :param y_pred: Prediction result of the data in one-hot-encoding type.
    :return: None.
    """
    # Calculate fpr, tpr, and area under the curve(auc)
    # Transform y_true and y_pred from one-hot-encoding to the label-encoding.
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot.
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  # plot roc curve
    plt.plot([0, 1], [0, 1], 'k--', color='navy', lw=2)  # plot random guess line

    # Set labels, title, ticks, legend, axis range and annotation.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.53, .48), color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig("../output/dataset-{}_model-{}_imagesize-{}_ROC-binary.png".format(config.dataset, config.model, config.imagesize))
    plt.show()


def plot_roc_curve_multiclass(y_true: list, y_pred: list, label_encoder) -> None:
    """
    Plot ROC curve for multi classification.

    Code reference: https://github.com/DeepmindHub/python-/blob/master/ROC%20Curve%20Multiclass.py

    :param y_true: Ground truth of the data in one-hot-encoding type.
    :param y_pred: Prediction result of the data in one-hot-encoding type.
    :return: None.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate fpr, tpr, area under the curve(auc) of each class.
    for i in range(label_encoder.classes_.size):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate macro fpr, tpr and area under the curve (AUC).
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(label_encoder.classes_))]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(label_encoder.classes_.size):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= label_encoder.classes_.size

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Calculate micro fpr, tpr and area under the curve (AUC).
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Plot.
    plt.figure(figsize=(8, 5))

    # Plot micro roc curve.
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', lw=4)

    # Plot macro roc curve.
    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['macro']),
             color='black', linestyle=':', lw=4)

    # Plot roc curve of each class.
    colors = ['#3175a1', '#e1812b', '#39923a', '#c03d3e', '#9372b2']
    for i, color in zip(range(len(label_encoder.classes_)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(label_encoder.classes_[i], roc_auc[i]))

    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', color='red', lw=2)

    # Set labels, title, ticks, legend, axis range and annotation.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.53, .48), color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig("../output/dataset-{}_model-{}_imagesize-{}_ROC-multi.png".format(config.dataset, config.model, config.imagesize))
    plt.show()


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
        plt.savefig("../output/dataset-{}_model-{}_imagesize-{}_CM-norm.png".format(config.dataset, config.model, config.imagesize))
    elif not is_normalised:
        plt.savefig("../output/dataset-{}_model-{}_imagesize-{}_CM.png".format(config.dataset, config.model, config.imagesize))
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
    plt.savefig("../output/dataset-{}_model-{}_imagesize-{}_{}.png".format(config.dataset, config.model, config.imagesize, title), bbox_inches='tight')
    plt.show()


def evaluate(y_true: list, y_pred: list, label_encoder: LabelEncoder, dataset: str, classification_type: str):
    """
    Evaluate model performance with accuracy, confusion matrix, ROC curve and compare with other papers' results.
    :param y_true: Ground truth of the data in one-hot-encoding type.
    :param y_pred: Prediction result of the data in one-hot-encoding type.
    :param label_encoder: The label encoder for y value (label).
    :param dataset: The dataset to use.
    :param classification_type: The classification type. Ex: N-B-M: normal, benign and malignant; B-M: benign and
    malignant.
    :return: None.
    """
    # Inverse transform y_true and y_pred from one-hot-encoding to original label.
    if label_encoder.classes_.size == 2:
        y_true_inv = y_true
        y_pred_inv = np.round_(y_pred, 0)
    else:
        y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
        y_pred_inv = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))

    # Calculate accuracy.
    accuracy = float('{:.4f}'.format(accuracy_score(y_true_inv, y_pred_inv)))
    print('accuracy = {}\n'.format(accuracy))

    # Print classification report for precision, recall and f1.
    print(classification_report(y_true_inv, y_pred_inv, target_names=label_encoder.classes_))

    # Plot confusion matrix and normalised confusion matrix.
    cm = confusion_matrix(y_true_inv, y_pred_inv)  # calculate confusion matrix with original label of classes
    plot_confusion_matrix(cm, 'd', label_encoder, False)
    # Calculate normalized confusion matrix with original label of classes.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0
    plot_confusion_matrix(cm_normalized, '.2f', label_encoder, True)

    # Plot ROC curve.
    if label_encoder.classes_.size == 2:  # binary classification
        plot_roc_curve_binary(y_true, y_pred)
    elif label_encoder.classes_.size >= 2:  # multi classification
        plot_roc_curve_multiclass(y_true, y_pred, label_encoder)

    # Compare our results with other papers' result.
    with open('data_visualisation/other_paper_results.json') as config_file:  # load other papers' result from json file
        data = json.load(config_file)
    df = pd.DataFrame.from_records(data[dataset][classification_type],
                                   columns=['paper', 'accuracy'])  # Filter data by dataset and classification type.
    new_row = pd.DataFrame({'paper': 'Dissertation', 'accuracy': accuracy},
                           index=[0])  # Add model result into dataframe to compare.
    df = pd.concat([new_row, df]).reset_index(drop=True)
    df['accuracy'] = pd.to_numeric(df['accuracy'])  # Digitize the accuracy column.
    plot_comparison_chart(df)


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
    plt.savefig("../output/dataset-{}_model-{}_imagesize-{}_{}.png".format(config.dataset, config.model, config.imagesize, plot_name))
    plt.show()
