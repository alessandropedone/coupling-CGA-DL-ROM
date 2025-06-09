import os
# Suppress TensorFlow debug/info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # "0" = all logs, "1" = filter INFO, "2" = filter INFO & WARNING, "3" = only ERRORs
# suppress XLA-related logs
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="  # Prevent CUDA libdevice warnings
# disable absl logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


from plot_prediction import plot_random_prediction
import tensorflow as tf
import pandas as pd
import numpy as np
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

    seed = 40

    # Split into train+val and test sets first
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )

    # Split train+val into train and val sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, random_state=seed
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
        l1_coeff= 0, 
        l2_coeff = 1e-3, 
        batch_normalization = True, 
        dropout = True, 
        dropout_rate = 0.2, 
        leaky_relu_alpha = None,
        layer_normalization = False,
        positional_encoding_frequencies = 8,
    )

    model.summary()

    def lr_warmup_schedule(epoch, lr):
        warmup_epochs = 5
        base_lr = 5e-4
        start_lr = 1e-6
        if epoch <= warmup_epochs:
            return start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
        return lr
    
    warmup_callback = tf.keras.callbacks.LearningRateScheduler(lr_warmup_schedule, verbose=0)

    reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

    model.train_model(
        X = x_train, 
        y = y_train, 
        X_val = x_val, 
        y_val = y_val, 
        learning_rate = 1e-3, 
        epochs = 1000, 
        batch_size = 2048, 
        loss = 'mse', 
        validation_freq = 1, 
        verbose = 1, 
        lr_scheduler = [warmup_callback, reduce_callback], 
        metrics = ['mae', 'mse'],
        clipnorm = 1, 
        early_stopping_patience = 15,
        log = True,
        optimizer = 'adam',
    )

    print("Evaluating the model on the validation set...")
    model.evaluate_model(X = x_val, y = y_val)

    print("Evaluating the model on the test set...")
    model.evaluate_model(X = x_test, y = y_test)

    model.save_model(model_path)
    

def test_model(model_path: str):
    plot_random_prediction(model_path)


def run_on_device(func, *args, **kwargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only (optional)

    # List GPUs and set memory growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass  # Already set or not supported

    device = '/GPU:0' if gpus else '/CPU:0'

    with tf.device(device):
        # Dummy op to check placement
        a = tf.constant([[1.0]])
        if 'GPU' in a.device:
            print("✅ Using GPU for execution")
        else:
            print("⚠️ Using CPU for execution")

        # Call the provided function with arguments
        return func(*args, **kwargs)

run_on_device(train_model, "models/model10.keras")
run_on_device(test_model, "models/model10.keras")


