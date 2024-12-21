# Define EarlyStopping callback
early_stopping = tfk.callbacks.EarlyStopping(
    monitor='val_loss',
    # Number of epochs with no improvement after which training will be stopped
    patience=2,
    # Restores model weights from the epoch with the best monitored metric
    restore_best_weights=True,
)
