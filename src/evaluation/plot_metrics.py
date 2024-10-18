import matplotlib.pyplot as plt


def plot_full_metrics(history):

    # Extract the metrics from the history object
    history_dict = history.history

    # accuracy = history_dict['accuracy']
    # val_accuracy = history_dict['val_accuracy']

    # Convert to percentage
    accuracy = [a * 100 for a in history_dict['accuracy']]
    val_accuracy = [va * 100 for va in history_dict['val_accuracy']]

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    recall = history_dict['recall']
    val_recall = history_dict['val_recall']
    precision = history_dict['precision']
    val_precision = history_dict['val_precision']

    # Define the range of epochs
    epochs = range(1, len(accuracy) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, recall, 'bo-', label='Training Recall')
    plt.plot(epochs, val_recall, 'ro-', label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 4)
    plt.plot(epochs, precision, 'bo-', label='Training Precision')
    plt.plot(epochs, val_precision, 'ro-', label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


def plot_simple_metrics(history):

    # Extract the metrics from the history object
    history_dict = history.history

    # accuracy = history_dict['accuracy']
    # val_accuracy = history_dict['val_accuracy']

    # Convert to percentage
    accuracy = [a * 100 for a in history_dict['accuracy']]
    val_accuracy = [va * 100 for va in history_dict['val_accuracy']]

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # Define the range of epochs
    epochs = range(1, len(accuracy) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
