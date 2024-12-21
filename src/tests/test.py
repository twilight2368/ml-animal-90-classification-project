import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

print(os.path.pardir)


# todo View our example image
# * !wget https: // raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg
# steak = mpimg.imread("03-steak.jpeg")
# plt.imshow(steak)
# plt.axis(False)

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor
    and reshapes it to (img_shape, img_shape, colour_channel).
    """
    # Read in target file (an image)
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size=[img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.

    # Add an extra axis
    print(f"Shape before new dimension: {img.shape}")
    img = tf.expand_dims(img, axis=0)  # add an extra dimension at axis 0
    # img = img[tf.newaxis, ...] # alternative to the above, '...' is short for 'every other dimension'
    print(f"Shape after new dimension: {img.shape}")

    return img


def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(img)

    print(pred)
    # Get the predicted class
    pred_class = class_names[int(tf.round(pred)[0][0])]

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
