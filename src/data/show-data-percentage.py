import os
import matplotlib.pyplot as plt
import numpy as np


def displayPercentageBetweenTrainAndValidation(PATH_DIR_train, PATH_DIR_val):
    total_image_training = 0

    total_image_validation = 0

    for dirpath, dirnames, filenames in os.walk(PATH_DIR_train):
        total_image_training += len(filenames)

    for dirpath, dirnames, filenames in os.walk(PATH_DIR_val):
        total_image_validation += len(filenames)

    print(f"Total training images: {total_image_training}")
    print(f"Total validation images: {total_image_validation}")

    print(
        f"Percentage of training data: {total_image_training/(total_image_training + total_image_validation)*100}%")
    print(
        f"Percentage of validation data: {total_image_validation/(total_image_training + total_image_validation)*100}%")

    plot_data = np.array([total_image_training, total_image_validation])

    plot_labels = ["training", "validation"]

    plt.pie(plot_data, labels=plot_labels, autopct='%1.1f%%')
    plt.title('Percentage of training and validation data')
