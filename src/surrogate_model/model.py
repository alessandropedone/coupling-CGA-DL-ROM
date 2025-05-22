import os
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Optional, Callable, Tuple
import numpy as np
import datetime


class NN_Model:
    """
    A class to build, train, and manage a neural network model using TensorFlow and Keras.
    """

    ##
    def __init__(self):
        """
        Initializes the NN_Model class with an empty Sequential model and a None history.
        """
        self.model: tf.keras.models.Sequential = tf.keras.models.Sequential()
        self.history = None

    ## 
    # @param model_path (str): The path to the saved model.
    # @return None
    def load_model(self, model_path: str) -> None:
        """
        Loads a model saved at the specified path.
        """
        self.model = tf.keras.models.load_model(model_path)

    ##
    # @param input_shape (int): The shape of the input layer.
    # @param n_neurons (List[int]): A list containing the number of neurons for each layer.
    # @param activation (str): The activation function for the hidden layers.
    # @param output_neurons (int): The number of neurons in the output layer.
    # @param output_activation (str): The activation function for the output layer.
    # @param initializer (str): The initializer for the layer weights.
    # @param lambda_coeff (float): The coefficient for L2 regularization.
    # @return None
    # @throws ValueError: If the length of `n_neurons` and `activations` lists do not match.
    def build_model(self,  
                    X : np.array,
                    input_shape: (int), 
                    n_neurons: List[int] = [64, 64, 64, 64, 64, 64, 64, 64], 
                    activation: str = 'tanh',
                    output_neurons: int = 1,
                    output_activation: str = 'linear',
                    initializer: str = 'glorot_uniform',
                    lambda_coeff: float = 1e-9,
                    batch_normalization: bool = False,
                    dropout: bool = False,
                    dropout_rate: float = 0.3) -> None:
        """
        Constructs the neural network model layer by layer.
        """

        l2 = tf.keras.regularizers.l2
        Dense = tf.keras.layers.Dense
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        Normalization = tf.keras.layers.Normalization

        self.model.add(tf.keras.layers.InputLayer(shape=(input_shape,)))

        normalizer = Normalization(axis=-1)
        normalizer.adapt(X)
        self.model.add(normalizer)

        self.model.add(Dense(n_neurons[0], activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(lambda_coeff)))    
        if batch_normalization:
            self.model.add(BatchNormalization())  
        if dropout:
            self.model.add(Dropout(dropout_rate))

        for neurons in n_neurons[1:]:
            self.model.add(Dense(neurons, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(lambda_coeff)))
            if batch_normalization:
                self.model.add(BatchNormalization())  
            if dropout:
                self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(output_neurons, activation=output_activation, kernel_regularizer=l2(lambda_coeff)))

    ##
    # @param X (np.ndarray): The input data for training.
    # @param y (np.ndarray): The target data for training.
    # @param X_val (np.ndarray): The input data for validation.
    # @param y_val (np.ndarray): The target data for validation.
    # @param learning_rate (float): The learning rate for the optimizer.
    # @param epochs (int): The number of epochs for training.
    # @param batch_size (int): The size of the batches for training.
    # @param loss (str): The loss function to be used during training.
    # @param validation_freq (int): The frequency of validation during training.
    # @param lr_schedule (Optional[Callable[[int], float]]): A function to adjust the learning rate.
    # @return None
    # @throws ValueError: If any of the input arrays are empty.
    def train_model(self, 
                    X: np.ndarray, 
                    y: np.ndarray, 
                    X_val: np.ndarray, 
                    y_val: np.ndarray, 
                    learning_rate: float = 1e-3, 
                    epochs: int = 10000, 
                    batch_size: int = 15000, 
                    loss: str = 'mean_squared_error', 
                    validation_freq: int = 1, 
                    verbose: int = 0,
                    lr_scheduler = None,
                    metrics: list = ['mse'],
                    clipnorm: float = None,
                    early_stopping_patience: int = None,
                    log: bool = False
                    ) -> None:
        """
        Trains the model on the provided dataset.
        Use "tensorboard --logdir logs" to visualize logs (if log is set to True).
        """
        if X.size == 0 or y.size == 0 or X_val.size == 0 or y_val.size == 0:
            raise ValueError("Input arrays must not be empty")

        Adam = tf.keras.optimizers.Adam
        if clipnorm is not None:
            self.model.compile(loss=loss, metrics=metrics,optimizer=Adam(learning_rate=learning_rate, clipnorm=clipnorm))
        else:
            self.model.compile(loss=loss, metrics=metrics,optimizer=Adam(learning_rate=learning_rate))
        
        callbacks = []
        if lr_scheduler is not None:
            callbacks.append(lr_scheduler)

        # Set up TensorBoard callback with profiling
        if log:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_steps_per_second=True)
            callbacks.append(tensorboard_callback)
        # TensorBoard command: tensorboard --logdir logs

        # Early stopping callback
        if early_stopping_patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        self.history = self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=(X_val, y_val), validation_freq=validation_freq,
            callbacks=callbacks
        )

        self.plot_training_history()

    ##
    def plot_training_history(self) -> None:
        """
        Plots the training and validation loss over epochs.
        This method should be called after training the model using `train_model`.
        """
        if self.history is None:
            raise ValueError("The model has no training history. Train the model using 'train_model' method first.")

        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        tot_train = len(self.history.history['loss'])
        tot_valid = len(self.history.history['val_loss']) 
        valid_freq = int(tot_train / tot_valid)
        plt.plot(np.arange(tot_train), self.history.history['loss'], 'b-', label='Training loss', linewidth=2)
        plt.plot(valid_freq * np.arange(tot_valid), self.history.history['val_loss'], 'r--', label='Validation loss', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.grid(True)
        plt.show()

    ##
    # @param model_path (str): The file path where the model should be saved.
    def save_model(self, model_path: str) -> None:
        """
        Saves the current state of the model to a specified file path.
        """
        self.model.save(model_path)

    ## 
    def evaluate_model(self,
                        X: np.ndarray, 
                        y: np.ndarray) -> None:
        results = self.model.evaluate(X, y, return_dict=True)
        print(results)

    ##
    def summary(self) -> None:
        self.model.summary(expand_nested=False, show_trainable=True)

    ##
    # @param X (np.ndarray): The input data for making predictions.
    # @return np.ndarray: The predicted values.
    # @returns The predicted values.
    # @throws ValueError: If the input array is empty.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model.
        """
        return self.model.predict(X, verbose=0)