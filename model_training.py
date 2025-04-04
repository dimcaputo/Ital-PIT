# Mistral Le Chat was used to add comments to this code.

# Import necessary libraries for data processing, model building, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.metrics import F1Score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


# Define a function to create a neural network model
def get_model(input_size, filters=8, dropout1=0.2, dropout2=0.2, classes=10):
    # Sequential model with multiple dense layers, dropout, and batch normalization
    model = Sequential([
        Input(shape=(input_size,)),  # Input layer
        Dense(filters, activation='relu'),  # Hidden layers with ReLU activation
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        BatchNormalization(),  # Normalization to stabilize learning
        Dropout(dropout1),  # Dropout to prevent overfitting
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        BatchNormalization(),
        Dropout(dropout2),
        Dense(filters, activation='relu'),
        Dense(classes, activation='softmax'),  # Output layer with softmax for classification
    ])
    print(model.summary())  # Display the model architecture
    return model

# Define a function for early stopping during training
def stop_early(patience=10, start=50):
    # Early stopping to halt training when validation performance stops improving
    return EarlyStopping(
        monitor='val_f1_score',  # Monitor F1 score on validation data
        patience=patience,  # Number of epochs to wait for improvement
        verbose=1,
        mode='max',  # Stop when F1 score stops increasing
        restore_best_weights=True,  # Restore best weights
        start_from_epoch=start  # Epoch to start monitoring
    )

# Load and preprocess data
df_land = pd.read_csv('training_data/landmarks.csv', index_col='pose_id')
df_angles = pd.read_csv('training_data/angles.csv', index_col='pose_id')
df_labels = pd.read_csv('training_data/labels.csv', index_col='pose_id')

df_features = pd.concat([df_land, df_angles], axis=1)

# Prepare data for training
X = df_features
y = pd.get_dummies(df_labels, dtype=int)  # One-hot encode labels
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Standardize features
X_test = scaler.transform(X_test)

# Create, compile, and train the model
model = get_model(X_train.shape[1], 64, 0.2, 0.2, 10)
model.compile(optimizer='adam', metrics=[F1Score(average='macro', name='f1_score')], loss='categorical_focal_crossentropy')

history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[stop_early(patience=50)])

# Evaluate the model
print(classification_report(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1), target_names=y.columns.values))


# Predict and map labels for the test set
df_res = pd.DataFrame(np.argmax(model.predict(X_test), axis=1))
df_res = df_res.map(lambda x: y.columns.values[x])



# Save the trained model
model.save('classifier.keras')

# Save the classes in order
with open("classes.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(y.columns.values.tolist())

vf1_arr = history.history["val_f1_score"]
f1_arr = history.history["f1_score"]
plt.plot(range(len(vf1_arr)), vf1_arr, label='Validation F1 Score')
plt.plot(range(len(f1_arr)), f1_arr, label='Training F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
