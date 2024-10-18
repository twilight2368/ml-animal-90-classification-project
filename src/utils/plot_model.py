from tensorflow.keras.utils import plot_model

# todo: Plot the model


def plot_my_model(model):
    plot_model(model, show_shapes=True, show_layer_names=True)


def plot_my_model_and_export(model, model_name):
    plot_model(model, to_file=model_name + '.png',
               show_shapes=True, show_layer_names=True)
