
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Resizing, Flatten
from keras.metrics import F1Score
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os


def stop_early(patience=10, start=50):
    # Early stopping to halt training when validation performance stops improving
    return EarlyStopping(
        monitor='val_accuracy',  # Monitor F1 score on validation data
        patience=patience,  # Number of epochs to wait for improvement
        verbose=1,
        mode='max',  # Stop when F1 score stops increasing
        restore_best_weights=True,  # Restore best weights
        start_from_epoch=start  # Epoch to start monitoring
    )


train_ds, valid_ds = image_dataset_from_directory(
    'images',
    image_size = (256,256),
    seed = 38,
    subset = 'both',
    validation_split = 0.2
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    Input(shape=(256,256,3,)),
    Resizing(256,256),
    Conv2D(16, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(32, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(64, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(16, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(32, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(64, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(16, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(32, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(64, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),    
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(16, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(32, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(64, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_ds,
    epochs=100,
    validation_data=valid_ds,
    callbacks=[stop_early()]
)

model.save('cnn.keras')