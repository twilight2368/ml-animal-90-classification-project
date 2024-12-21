import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import tf_keras as tfk
import sklearn
import pandas as pd
import matplotlib
import numpy as np
import tensorflow as tf
from pathlib import Path
import os.path
import matplotlib.image as mpimg
import os


from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

# Please run this before start using the notebook

import datetime
print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print(f"TensorFlow version: {tf.__version__}")

print(f"NumPy version: {np.__version__}")

print(f"Matplotlib version: {matplotlib.__version__}")

print(f"Pandas version: {pd.__version__}")

print(f"Scikit-Learn version: {sklearn.__version__}")

print(f"TensorFlow Keras version: {tfk.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")

