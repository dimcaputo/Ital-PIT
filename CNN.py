
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Resizing, Flatten
from keras.metrics import F1Score
from keras.utils import image_dataset_from_directory, to_categorical
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import csv
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import regex as re
import pandas as pd


def stop_early(patience=10, start=10):
    # Early stopping to halt training when validation performance stops improving
    return EarlyStopping(
        monitor='val_f1_score',  # Monitor F1 score on validation data
        patience=patience,  # Number of epochs to wait for improvement
        verbose=1,
        mode='max',  # Stop when F1 score stops increasing
        restore_best_weights=True,  # Restore best weights
        start_from_epoch=start  # Epoch to start monitoring
    )

def make_arrays(filepath, size):
    labels = []
    file_count = sum(len(files) for _, _, files in os.walk(filepath))
    for root, dirs, files in os.walk(filepath):
        array = np.zeros(shape=(file_count,size,size,3))
        for i, file in enumerate(files):
            with Image.open(os.path.join(root, file)) as im:
                array[i,:,:,:] = im.resize((size, size), Image.Resampling.LANCZOS)
                labels.append(re.sub(r'^.+/','', root))
        print(f'{root} folder done')
    labels = np.asarray(labels)
    return array, labels

size = 128
X, y = make_arrays('images/', size)

y = pd.get_dummies(y, dtype=int)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=38)

filters = 16

model = Sequential([
    Input(shape=(size,size,3,)),
    Conv2D(filters, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*2, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*4, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(filters, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*2, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*4, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(filters, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*2, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*4, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),   
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(filters, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*2, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*4, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(filters, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*2, kernel_size= 3, activation='relu', padding='same'),
    Conv2D(filters*4, kernel_size= 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[F1Score(average='macro', name='f1_score')])
model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.1,
    callbacks=[stop_early(10,50)],
    batch_size=64
)

model.save('cnn.keras')