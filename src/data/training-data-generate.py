import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# todo:  Create a dataset from the directory


def generate_training_data_from_directory(PATH_DIR, batch_size=32, img_height=224, img_width=224):

    # Load the dataset
    train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        PATH_DIR,
        image_size=(img_height, img_width),  # Resize all images to this size
        batch_size=batch_size,
        label_mode='categorical',  # 'categorical' for one-hot encoding
        shuffle=True,  # Shuffle the dataset
        seed=42  # Set a seed for reproducibility
    )

    val_generator = tf.keras.preprocessing.image_dataset_from_directory(
        PATH_DIR,
        image_size=(img_height, img_width),  # Resize all images to this size
        batch_size=batch_size,
        label_mode='categorical',  # 'categorical' for one-hot encoding
        shuffle=False,  # Shuffle the dataset
        seed=42  # Set a seed for reproducibility
    )

    print("========================================================")
    # Verify the dataset
    training_class_names = train_generator.class_names
    print("Training classnames: ", training_class_names)
    print("========================================================")
    validation_class_names = val_generator.class_names
    print("Validation classnames: ", validation_class_names)

    return train_generator, val_generator

# todo: Normalize the images


def normalize_train_data(training_data, validation_data):

    # todo: normalize images:
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    x_data = training_data.map(normalize_img)

    y_data = validation_data.map(normalize_img)

    return x_data, y_data


def generate_training_data_from_dataframe(train_df, val_df, batch_size=32, img_height=224, img_width=224):

    # Define the data generators
    train_datagen = ImageDataGenerator(rescale=1./255)

    # Only rescale for validation and test data
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create the generators for training, validation, and test sets
    train_generator = train_datagen.flow_from_dataframe(
        # This assumes you split the data into 'train_df', 'val_df', etc.
        train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        batch_size=30,
        class_mode='categorical',
        shuffle=False,
    )

    return train_generator, val_generator
