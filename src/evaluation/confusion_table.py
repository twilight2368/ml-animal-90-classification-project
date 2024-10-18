import seaborn as sn
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sn


def create_confusion_table(model, validation_data):
    # Assume your model is already defined and trained
    # model = ...

    # Get the true labels from the validation dataset
    true_labels = []
    for images, labels in validation_data:
        true_labels.extend(np.argmax(labels.numpy(), axis=1))

    # Make predictions on the validation dataset
    predictions = model.predict(validation_data)
    predicted_labels = np.argmax(predictions, axis=1)

    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        np.array(true_labels),
        predicted_labels,
        rownames=['Actual'],
        colnames=['Predicted']
    )

    confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    print("Heatmap\n")
    plt.figure(figsize=(30, 30))
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues')
    plt.show()
