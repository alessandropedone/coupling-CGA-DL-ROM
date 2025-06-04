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
    y_pred = model.predict(x)
    y_pred = y_pred.flatten()

    # Plot the prediction
    # Take x coordinates from the 4th column
    coords = x[:, 3]
    # Sort the coordinates and corresponding y and y_pred values
    sorted_indices = np.argsort(coords)
    coords = coords[sorted_indices]
    y = y[sorted_indices]
    y_pred = y_pred[sorted_indices]

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(coords, y, label="Normal derivative values", color="blue", linestyle="-")
    plt.plot(coords, y_pred, label=f"Model prediction", color="red", linestyle="--")
    plt.xlim(-50, 50)
    plt.xlabel("coords")
    plt.ylabel("normal_derivative_potential")
    plt.title("Surrogate model prediction test")
    plt.legend()
    plt.grid(True)
    #plt.show()

    # Plot the absolute difference between prediction and true values
    abs_diff = np.abs(y - y_pred)
    mse = np.mean((y - y_pred) ** 2)
    plt.figure(2)
    plt.plot(coords, abs_diff, label="Absolute Error", color="green")
    plt.xlim(-50, 50)
    plt.xlabel("coords")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error of Surrogate Model Prediction (MSE: {mse:.4f})")
    mean_abs_error = np.mean(abs_diff)
    plt.axhline(mean_abs_error, color="orange", linestyle="--", label=f"Mean Abs Error: {mean_abs_error:.4f}")
    plt.axhline(np.sqrt(mse), color="purple", linestyle="--", label=f"RMSE: {np.sqrt(mse):.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()

##
# @param model_path (str): The path to the saved model.
def plot_random_prediction(model_path: str):
    """
    Plot a random prediction from the surrogate model.
    """
    # Import parameters, coordinates and normal derivative dataset
    data = pd.read_csv('data/unrolled_normal_derivative_potential.csv')

    # Save parameters and coordinates and convert to numpy array
    x = data.iloc[:, 1:5]
    x = x.to_numpy()

    # Save normal derivative potential values and convert to numpy array
    y = data.iloc[:, 5]
    y = y.to_numpy()

    # Predict on a random test element
    # Select a random test element
    import random

    # Select random combination of the 3 geometric parameters
    random_index = random.randint(0, len(x) - 1)
    triplet = x[random_index, 0:3]

    # Now retrieve all the coordinates that match this triplet
    mask = np.all(x[:, 0:3] == triplet, axis=1)

    x_sample = x[mask]
    y_sample = y[mask]

    plot_prediction(x_sample, y_sample, model_path=model_path)