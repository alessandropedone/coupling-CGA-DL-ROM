import os
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Optional, Callable, Tuple
import numpy as np
import datetime
from keras.saving import register_keras_serializable


@register_keras_serializable()
class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, positional_encoding_frequencies, **kwargs):
        super().__init__(**kwargs)
        self.positional_encoding_frequencies = positional_encoding_frequencies

    def call(self, x):
        x3 = tf.expand_dims(x[:, 3], -1)
        encoded = [x3]
        for i in range(1, self.positional_encoding_frequencies + 1):
            freq = 2.0 ** i * np.pi
            encoded.append(tf.sin(freq * x3))
            encoded.append(tf.cos(freq * x3))
        return tf.concat([x[:, :3], *encoded], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "positional_encoding_frequencies": self.positional_encoding_frequencies
        })
        return config
    
@register_keras_serializable()
class FourierFeatures(tf.keras.layers.Layer):
    def __init__(self, num_frequencies, learnable=True, initializer='glorot_uniform', **kwargs):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.learnable = learnable
        self.initializer = initializer

    def build(self, input_shape):
        shape = (1, self.num_frequencies)
        if self.learnable:
            self.freqs = self.add_weight(name="freqs", shape=shape,
                                         initializer=self.initializer,
                                         trainable=True)
        else:
            self.freqs = tf.constant(2.0 ** tf.range(1, self.num_frequencies + 1, dtype=tf.float32)[tf.newaxis, :])

    def call(self, x):
        # Use the same approach as PositionalEncodingLayer, but with learnable or fixed frequencies
        x3 = tf.expand_dims(x[:, 3], -1)
        encoded = [x3]
        for i in range(self.num_frequencies):
            freq = self.freqs[0, i]
            encoded.append(tf.sin(freq * x3))
            encoded.append(tf.cos(freq * x3))
        return tf.concat([x[:, :3], *encoded], axis=-1)
    

class LogUniformFreqInitializer(tf.keras.initializers.Initializer):
    def __init__(self, min_exp=0.0, max_exp=8.0):
        self.min_exp = min_exp
        self.max_exp = max_exp
        
    def __call__(self, shape, dtype=None):
        # Sample uniformly from [min_exp, max_exp]
        exponents = tf.random.uniform(shape, self.min_exp, self.max_exp, dtype=dtype)
        return tf.math.pow(2.0, exponents)

    def get_config(self):
        return {'min_exp': self.min_exp, 'max_exp': self.max_exp}


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
    # @param l1_coeff (float): The coefficient for L1 regularization.
    # @param l2_coeff (float): The coefficient for L2 regularization.
    # @param batch_normalization (bool): Whether to apply batch normalization after each layer.
    # @param dropout (bool): Whether to apply dropout after each layer.
    # @param dropout_rate (float): The dropout rate to be applied if dropout is True.
    # @param layer_normalization (bool): Whether to apply layer normalization after each layer.
    # @param positional_encoding_frequencies (int): The number of frequencies for positional encoding.
    # @return None
    # @throws ValueError: If the length of `n_neurons` and `activations` lists do not match.
    def build_model(self,  
                X: np.ndarray,
                input_shape: int, 
                n_neurons: list = [64, 64, 64, 64, 64, 64, 64, 64], 
                activation: str = 'tanh',
                output_neurons: int = 1,
                output_activation: str = 'linear',
                initializer: str = 'glorot_uniform',
                l1_coeff: float = 0,
                l2_coeff: float = 0,
                batch_normalization: bool = False,
                dropout: bool = False,
                dropout_rate: float = 0.3,
                leaky_relu_alpha: float = None,
                layer_normalization: bool = False,
                positional_encoding_frequencies: int = 0) -> None:
        """
        Constructs the neural network model layer by layer with optional positional encoding.
        """

        l1_l2 = tf.keras.regularizers.l1_l2
        Dense = tf.keras.layers.Dense
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        LeakyReLU = tf.keras.layers.LeakyReLU
        Normalization = tf.keras.layers.Normalization

        inputs = tf.keras.Input(shape=(input_shape,))

        normalizer = Normalization(axis=-1)
        normalizer.adapt(X)
        x = normalizer(inputs)

        x = PositionalEncodingLayer(positional_encoding_frequencies=positional_encoding_frequencies)(x)

        #x = FourierFeatures(num_frequencies=positional_encoding_frequencies, learnable=True, initializer=LogUniformFreqInitializer(min_exp=0.0, max_exp=8.0))(x)
        
        # First layer
        if leaky_relu_alpha is not None:
            x = Dense(n_neurons[0], kernel_initializer=initializer,
                    kernel_regularizer=l1_l2(l1=l1_coeff, l2=l2_coeff))(x)
            x = LeakyReLU(alpha=leaky_relu_alpha)(x)
        else:
            x = Dense(n_neurons[0], activation=activation,
                    kernel_initializer=initializer,
                    kernel_regularizer=l1_l2(l1=l1_coeff, l2=l2_coeff))(x)

        if batch_normalization:
            x = BatchNormalization()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

        # Hidden layers
        for neurons in n_neurons[1:]:
            if leaky_relu_alpha is not None:
                x = Dense(neurons, kernel_initializer=initializer,
                        kernel_regularizer=l1_l2(l1=l1_coeff, l2=l2_coeff))(x)
                x = LeakyReLU(alpha=leaky_relu_alpha)(x)
            else:
                x = Dense(neurons, activation=activation,
                        kernel_initializer=initializer,
                        kernel_regularizer=l1_l2(l1=l1_coeff, l2=l2_coeff))(x)

            if batch_normalization:
                x = BatchNormalization()(x)
            if dropout:
                x = Dropout(dropout_rate)(x)
            if layer_normalization:
                x = tf.keras.layers.LayerNormalization()(x)

        # Output layer
        if leaky_relu_alpha is not None:
            x = Dense(output_neurons,
                    kernel_regularizer=l1_l2(l1=l1_coeff, l2=l2_coeff))(x)
            x = LeakyReLU(alpha=leaky_relu_alpha)(x)
        else:
            x = Dense(output_neurons, activation=output_activation,
                    kernel_regularizer=l1_l2(l1=l1_coeff, l2=l2_coeff))(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)


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
    # @param optimizer (str): The optimizer to be used for training. Options are 'adam', 'sgd', 'rmsprop'.
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
                    log: bool = False,
                    optimizer: str = 'adam'
                    ) -> None:
        """
        Trains the model on the provided dataset.
        Use "tensorboard --logdir logs" to visualize logs (if log is set to True).
        """
        if X.size == 0 or y.size == 0 or X_val.size == 0 or y_val.size == 0:
            raise ValueError("Input arrays must not be empty")
        if loss == 'huber_loss':
            loss = tf.keras.losses.Huber(delta=1.0)

        if optimizer not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError("Unsupported optimizer. Supported optimizers are: 'adam', 'sgd', 'rmsprop'.")
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD
            if clipnorm is not None:
                self.model.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True, clipnorm=clipnorm))
            else:
                self.model.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True))
        elif optimizer == 'rmsprop':
            self.model.compile(loss=loss, metrics=metrics,optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False))
        elif optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam
            if clipnorm is not None:
                self.model.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, clipnorm=clipnorm))
            else:
                self.model.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate))
        
        callbacks = []
        if lr_scheduler is not None:
            for callback in lr_scheduler:
                callbacks.append(callback)

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