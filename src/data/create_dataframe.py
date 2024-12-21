from pathlib import Path
import os.path


def create_data_frame_from_image_path(PATH_DIR):
    image_dir = Path(PATH_DIR)
    filepaths = list(image_dir.glob(r'**/*.jpg'))

    labels = list(map(lambda x: os.path.split(
        os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Combine filepaths and labels into a DataFrame
    images = pd.concat([filepaths, labels], axis=1)

    # Shuffle the data and reset the index
    image_df = images.sample(frac=1.0, random_state=1).reset_index(drop=True)

    # Display the final DataFrame
    return image_df


def split_data_train_val(total_class, image_df, val_size=1/2):

    # Assuming 'image_df' contains all the data with 'Filepath' and 'Label' columns
    train_df, val_df = train_test_split(
        image_df, test_size=val_size, random_state=1, stratify=image_df['Label'])

    # Print the number of samples in each set
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    print(f"Each class training set: {len(train_df)/total_class} samples")
    print(f"Each class validation set: {len(val_df)/total_class} samples")

    plot_data = np.array([len(train_df), len(val_df)])

    plot_labels = ["training", "validation"]

    plt.pie(plot_data, labels=plot_labels, autopct='%1.1f%%')
    plt.title('Percentage of training, validation data')

    return train_df, val_df


def split_data_train_val_test(total_class, image_df, train_size=1/2, test_val_size=1/2):

    # Assuming 'image_df' contains all the data with 'Filepath' and 'Label' columns
    train_df, val_df = train_test_split(
        image_df, test_size=1-train_size, random_state=1, stratify=image_df['Label'])

    val_df, test_df = train_test_split(
        image_df, test_size=test_val_size, random_state=1, stratify=image_df['Label'])

    # Print the number of samples in each set
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    print(f"Each class training set: {len(train_df)/total_class} samples")
    print(f"Each class validation set: {len(val_df)/total_class} samples")

    plot_data = np.array([len(train_df), len(val_df)], len(test_df))

    plot_labels = ["training", "validation", "Testing"]

    plt.pie(plot_data, labels=plot_labels, autopct='%1.1f%%')
    plt.title('Percentage of training, validation, testing data')

    return train_df, val_df, test_df
