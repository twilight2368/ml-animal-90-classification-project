# View an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from PIL import Image


def view_random_image(target_dir, target_class):
    # Setup target directory (we'll view images from here)
    target_folder = target_dir+target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Image shape: {img.shape}")  # show the shape of the image

    return img


def view_a_random_images_class(target_dir, target_class):
    print(f"Image class: {target_class}")
    # Create the figure and set the overall title
    # Adjust figure size as needed
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    # Set the title for the entire figure
    fig.suptitle(target_class, fontsize=16)

    target_folder = os.path.join(target_dir, target_class)
    random_images = random.sample(os.listdir(
        target_folder), 4)  # Select 4 random images

    # Loop through each subplot to display the images
    for i in range(4):
        img = mpimg.imread(os.path.join(target_folder, random_images[i]))
        axs[i].imshow(img)
        axs[i].axis("off")  # Hide axes for each subplot

    plt.show()


def view_random_image_resize(target_dir, target_class, height=224, width=224):
    # Setup target directory (we'll view images from here)
    target_folder = target_dir + target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)[0]

    # Load the image using PIL and resize it to 224x224
    img = Image.open(os.path.join(target_folder, random_image))
    img = img.resize((height, width))  # Resize to 224x224 pixels

    return img


def view_a_random_images_class_resize(target_dir, target_class, height=224, width=224):
    print(f"Image class: {target_class}")
    # Create the figure and set the overall title
    # Adjust figure size as needed
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    # Set the title for the entire figure
    fig.suptitle(target_class, fontsize=16)

    target_folder = os.path.join(target_dir, target_class)
    random_images = random.sample(os.listdir(
        target_folder), 4)  # Select 4 random images

    # Loop through each subplot to display the images
    for i in range(4):
        img_path = os.path.join(target_folder, random_images[i])
        img = Image.open(img_path).resize((height, width))
        axs[i].imshow(img)
        axs[i].axis("off")  # Hide axes for each subplot

    plt.show()
