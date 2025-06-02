from plot_prediction import plot_random_prediction
import tensorflow as tf
import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from model import NN_Model

def train_model(model_path: str):
    # Import coordinates dataset and convert to numpy array
    coordinates = pd.read_csv('data/unrolled_normal_derivative_potential.csv')
    x = coordinates.iloc[:, 1:5]
    x = x.to_numpy()

    # Import normal derivative potential dataset and convert to numpy array
    normal_derivative = pd.read_csv('data/unrolled_normal_derivative_potential.csv')
    y = normal_derivative.iloc[:, 5]
    y = y.to_numpy()

    # Split into train+val and test sets first
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42
    )

    # Split train+val into train and val sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.15, random_state=42
    )

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Mixed Precision Setup
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model = NN_Model()

    model.build_model(
        X = x_train,
        input_shape = 4, 
        n_neurons = [512, 256, 128, 256], 
        activation = 'relu', 
        output_neurons = 1, 
        output_activation = 'linear', 
        initializer = 'he_normal', 
        lambda_coeff = 1e-3, 
        batch_normalization = True, 
        dropout = True, 
        dropout_rate = 0.2 
    )

    model.summary()

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    model.train_model(
        X = x_train, 
        y = y_train, 
        X_val = x_val, 
        y_val = y_val, 
        learning_rate = 1e-3, 
        epochs = 1000, 
        batch_size = 16, 
        loss = 'mean_squared_error', 
        validation_freq = 1, 
        verbose = 1, 
        lr_scheduler = lr_scheduler, 
        metrics = ['mean_absolute_error'],
        clipnorm = None,
        early_stopping_patience = 20,
        log = True
    )

    print("Evaluating the model on the validation set...")
    model.evaluate_model(X = x_val, y = y_val)

    print("Evaluating the model on the test set...")
    model.evaluate_model(X = x_test, y = y_test)

    model.save_model(model_path)
    

def test_model(model_path: str):
    plot_random_prediction(model_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify which GPU to use
gpus = tf.config.list_physical_devices('GPU')
#with tf.device('/GPU:0' if gpus else '/CPU:0'):
with tf.device('/CPU:0'):
    # Check if the GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Using GPU for training and evaluation")
    else:
        print("No GPU detected, using CPU")
    #train_model("models/new_model.keras")
    test_model("models/new_model_colab.keras")

