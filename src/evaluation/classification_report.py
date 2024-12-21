# Assuming `model` is your trained model and `validation_data` is a data generator
import numpy as np
from sklearn.metrics import classification_report


def classification_report_final(model, validation_data):
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

    # Generate the classification report
    class_labels = list(validation_data.class_indices.keys()
                        )      # Get class names
    report = classification_report(
        true_labels, predicted_labels, target_names=class_labels)
    print(report)
