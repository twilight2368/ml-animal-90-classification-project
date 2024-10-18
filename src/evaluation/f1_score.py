import tensorflow as tf

def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)  # Round predictions to get binary values
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    possible_positives = tf.reduce_sum(y_true)

    precision = true_positives / \
        (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision +
                                     recall + tf.keras.backend.epsilon())
    return f1
