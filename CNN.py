
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Resizing, Flatten
from keras.metrics import F1Score
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import csv
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


def stop_early(patience=10, start=10):
    # Early stopping to halt training when validation performance stops improving
    return EarlyStopping(
        monitor='val_accuracy',  # Monitor F1 score on validation data
        patience=patience,  # Number of epochs to wait for improvement
        verbose=1,
        mode='max',  # Stop when F1 score stops increasing
        restore_best_weights=True,  # Restore best weights
        start_from_epoch=start  # Epoch to start monitoring
    )

def make_arrays(filepath):
    labels = []
    for root, dirs, files in os.walk(filepath):
        array = np.zeros(shape=(len(files),360,480,3))
        for i, dir,file in enumerate(zip(dirs,files)):
            labels.append(dir)
            with Image.open(os.path.join(root, file)) as im:
                array[i,:,:,:] = im
    labels = np.asarray(labels)
    return array, labels


# train_ds, valid_ds = image_dataset_from_directory(
#     'images',
#     image_size = (128,128),
#     seed = 38,
#     subset = 'both',
#     validation_split = 0.2
# )

# class_names = train_ds.class_names

# with open("classes.csv", "w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(class_names)

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    Input(shape=(128,128,3,)),
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
    epochs=5,
    validation_data=valid_ds
)

model.save('cnn.keras')