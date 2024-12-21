import seaborn as sn
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sn


# def create_confusion_table(model, validation_data, steps_per_epoch):

#     # Get the true labels from the validation dataset
#     true_labels = []
#     for images, labels in validation_data:
#         true_labels.extend(np.argmax(labels, axis=1))

#     # Make predictions on the validation dataset
#     predictions = model.predict(validation_data, steps=steps_per_epoch, verbose=1)
#     predicted_labels = np.argmax(predictions, axis=1)

#     # Create confusion matrix
#     confusion_matrix = pd.crosstab(
#         np.array(true_labels),
#         predicted_labels,
#         rownames=['Actual'],
#         colnames=['Predicted']
#     )

#     return confusion_matrix

def create_confusion_table(model, validation_data):
    # Get the total number of samples and input image shape
    num_samples = validation_data.samples  # Total number of samples
    input_shape = validation_data.image_shape  # Shape of a single input image

    # Get the number of classes from the generator's class_indices
    num_classes = len(validation_data.class_indices)  # Number of classes

    # Allocate arrays for all data (images and labels)
    all_images = np.zeros((num_samples,) + input_shape, dtype=np.float32)
    all_labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Accumulate all images and labels
    current_index = 0  # Track current insertion point

    for batch_images, batch_labels in validation_data:
        batch_size = batch_images.shape[0]
        end_index = current_index + batch_size

        # Insert batch data into the allocated arrays
        all_images[current_index:end_index] = batch_images
        all_labels[current_index:end_index] = batch_labels

        current_index = end_index  # Move the pointer

        if end_index >= num_samples:  # Stop when all samples are processed
            break

    # Make predictions on the entire dataset at once
    predictions = model.predict(
        all_images, steps=len(validation_data), verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # Get the true labels from the accumulated labels
    true_labels = np.argmax(all_labels, axis=1)

    # Create confusion matrix using Pandas
    confusion_matrix = pd.crosstab(
        true_labels, predicted_labels, rownames=['Actual'], colnames=['Predicted']
    )

    return confusion_matrix


def create_confusion_table_with_class_names(model, validation_data):
    # Get the total number of samples and input image shape
    num_samples = validation_data.samples
    input_shape = validation_data.image_shape

    # Get the number of classes and class labels from class_indices
    class_indices = validation_data.class_indices
    # Map indices to class names
    class_labels = {v: k for k, v in class_indices.items()}

    num_classes = len(class_indices)

    # Allocate arrays for all data (images and labels)
    all_images = np.zeros((num_samples,) + input_shape, dtype=np.float32)
    all_labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Accumulate all images and labels
    current_index = 0

    for batch_images, batch_labels in validation_data:
        batch_size = batch_images.shape[0]
        end_index = current_index + batch_size

        # Insert batch data into the allocated arrays
        all_images[current_index:end_index] = batch_images
        all_labels[current_index:end_index] = batch_labels

        current_index = end_index

        if end_index >= num_samples:
            break

    # Make predictions on the entire dataset at once
    predictions = model.predict(
        all_images, steps=len(validation_data), verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # Get the true labels from the accumulated labels
    true_labels = np.argmax(all_labels, axis=1)

    # Map the indices in true_labels and predicted_labels to class names
    true_class_names = [class_labels[label] for label in true_labels]
    predicted_class_names = [class_labels[label] for label in predicted_labels]

    # Create confusion matrix using Pandas with class names
    confusion_matrix = pd.crosstab(
        pd.Series(true_class_names, name='Actual'),
        pd.Series(predicted_class_names, name='Predicted')
    )

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    print("Heatmap\n")
    plt.figure(figsize=(30, 30))
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues')
    plt.show()
