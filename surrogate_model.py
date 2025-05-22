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

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Mixed Precision Setup
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

gpus = tf.config.list_physical_devices('GPU')
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # Check if the GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Using GPU for training and evaluation")
    else:
        print("No GPU detected, using CPU")
    
    # Define the model
    # Alias for convenience
    layers = tf.keras.layers  
    regularizers = tf.keras.regularizers
    initializer = 'he_normal'
    regularizer = regularizers.l2(1e-4)
    norm_layer = layers.Normalization(axis=-1)
    norm_layer.adapt(x_train)
    model = tf.keras.Sequential([
        layers.InputLayer(shape=(353,)),
        norm_layer,
        layers.Dense(512, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(350)  # No activation, outputs can range from -inf to +inf
    ])
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    loss = tf.keras.losses.Huber()
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Set up TensorBoard callback with profiling
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_steps_per_second=True)

    # Add EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    )

    # Add ReduceLROnPlateau callback
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1
    )
    
    callbacks = [tensorboard_callback, early_stopping, lr_schedule]

    # Train the model with real data (increased batch size)
    batch_size = 32
    model.fit(
        x_train, y_train,
        epochs=500, 
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(x_val, y_val)
    )

    # Evaluate the model
    print("Evaluating the model on the validation set...")
    val_loss, val_accuracy = model.evaluate(x_val, y_val)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    print("Evaluating the model on the test set...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

# Save the model
model.save('surrogate_model.keras')

# TensorBoard command: tensorboard --logdir logs