import tensorflow as tf

# todo:  Create a dataset from the directory


def generate_training_data(batch_size=32, img_height=224, img_width=224):

    # Load the dataset
    X_train_model_1 = tf.keras.preprocessing.image_dataset_from_directory(
        PATH_DIR + 'train',
        image_size=(img_height, img_width),  # Resize all images to this size
        batch_size=batch_size,
        label_mode='categorical',  # 'categorical' for one-hot encoding
        shuffle=True,  # Shuffle the dataset
        seed=42  # Set a seed for reproducibility
    )

    y_validation_model_1 = tf.keras.preprocessing.image_dataset_from_directory(
        PATH_DIR + 'validation',
        image_size=(img_height, img_width),  # Resize all images to this size
        batch_size=batch_size,
        label_mode='categorical',  # 'categorical' for one-hot encoding
        shuffle=True,  # Shuffle the dataset
        seed=42  # Set a seed for reproducibility
    )

    print("========================================================")
    # Verify the dataset
    training_class_names = X_train_model_1.class_names
    print("Training classnames: ", class_names)
    print("========================================================")
    validation_class_names = y_validation_model_1.class_names
    print("Validation classnames: ", class_names)

    return training_class_names, validation_class_names

# todo: Normalize the images


def normalize_train_data(training_data, validation_data):

    # todo: normalize images:
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    x_data = training_data.map(normalize_img)

    y_data = validation_data.map(normalize_img)

    return x_data, y_data
