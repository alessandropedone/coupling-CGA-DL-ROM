import tensorflow as tf
import pandas as pd
import numpy as np
from model import NN_Model

##
# @param x (numpy.ndarray): The input data for the model.
# @param y (numpy.ndarray): The target data for the model.
# @param model_path (str): The path to the saved model.
def plot_prediction(x: np.ndarray, y: np.ndarray, model_path: str) -> None:
    """"
    Plot the prediction from the surrogate model.
    """
    # Load the saved model
    model = NN_Model()
    model.load_model(model_path)

    # Make a prediction
    y_pred = model.predict(np.expand_dims(x, axis=0))
    y_pred = y_pred.flatten()

    # Plot the prediction
    coords = x[3:]
    import matplotlib.pyplot as plt
    plt.plot(coords, y, label="Normal derivative values", color="blue", linestyle="-")
    plt.plot(coords, y_pred, label=f"Model prediction", color="red", linestyle="--")
    plt.xlim(-50, 50)
    plt.xlabel("coords")
    plt.ylabel("normal_derivative_potential")
    plt.title("Surrogate model prediction test")
    plt.legend()
    plt.grid(True)
    plt.show()

##
# @param model_path (str): The path to the saved model.
def plot_random_prediction(model_path: str):
    """
    Plot a random prediction from the surrogate model.
    """
    # Import coordinates dataset and convert to numpy array
    coordinates = pd.read_csv('data/coordinates.csv')
    x = coordinates.iloc[:, 1:]
    x = x.to_numpy()

    # Import normal derivative potential dataset and convert to numpy array
    normal_derivative = pd.read_csv('data/normal_derivative_potential.csv')
    y = normal_derivative.iloc[:, 4:]
    y = y.to_numpy()

    # Predict on a random test element
    # Select a random test element
    import random
    random_index = random.randint(0, len(x) - 1)
    x_sample = x[random_index]
    y_sample = y[random_index]

    plot_prediction(x_sample, y_sample, model_path=model_path)