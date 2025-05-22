import tensorflow as tf
import datetime
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify which GPU to use

# Import coordinates dataset and convert to numpy array
coordinates = pd.read_csv('data/coordinates.csv')
x = coordinates.iloc[:, 1:]
x = x.to_numpy()

# Import normal derivative potential dataset and convert to numpy array
normal_derivative = pd.read_csv('data/normal_derivative_potential.csv')
y = normal_derivative.iloc[:, 4:]
y = y.to_numpy()

# Split into train+val and test sets first
x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42
)

# Split train+val into train and val sets
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=0.15, random_state=42
)

# Load the saved model
loaded_model = tf.keras.models.load_model('surrogate_model.keras')

# Predict on a random test element
x_sample = x_test[0]
y_true = y_test[0]
y_pred = loaded_model.predict(np.expand_dims(x_sample, axis=0))
y_pred = y_pred.flatten()
coords = x_sample[3:]
import matplotlib.pyplot as plt
plt.plot(coords, y_true, label="Normale derivative values", color="blue", linestyle="-")
plt.plot(coords, y_pred, label=f"Model prediction", color="red", linestyle="--")
plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("normal_derivative_potential")
plt.title("Surrogate model prediction test")
plt.legend()
plt.grid(True)
plt.show()