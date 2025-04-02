from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.metrics import F1Score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_model(input_size, filters=8, dropout1=0.2, dropout2=0.2, classes=10):
    model = Sequential([
        Input(shape=(input_size,)),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        BatchNormalization(),
        Dropout(dropout1),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        Dense(filters, activation='relu'),
        BatchNormalization(),
        Dropout(dropout2),
        Dense(classes, activation='softmax'),
    ])
    model.summary()

    return model

def stop_early(patience=10, start=50):
    earlystopping=EarlyStopping(monitor='val_f1_score',
                                patience=patience,
                                verbose=1,
                                mode='max',
                                restore_best_weights=True,
                                start_from_epoch=start)
    return earlystopping


df_features = pd.read_csv(filepath_or_buffer='training_data/landmarks.csv',
                 index_col='pose_id')

df_labels = pd.read_csv(filepath_or_buffer='training_data/labels.csv',
                        index_col='pose_id')

df_no_poses = pd.read_csv(filepath_or_buffer='training_data/pose_landmarks_per_pose.csv').drop('Frame', axis=1)

df_no_poses['pose'] = 'no_pose'
no_pose_labels = df_no_poses.pose
df_no_poses=df_no_poses.drop('pose', axis=1)

df_no_poses.columns = df_features.columns
df_features = pd.concat([df_features, df_no_poses], axis=0).reset_index(drop=True)

df_labels = pd.concat([df_labels, no_pose_labels], axis=0).reset_index(drop=True)

X = df_features
y = pd.get_dummies(df_labels, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model2 = get_model(99, 64, 0.5, 0.5, 11)

model2.compile(optimizer='adam', metrics=[F1Score(average='macro', name='f1_score')], loss='categorical_focal_crossentropy')
history2 = model2.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[stop_early(patience=50)])

print(classification_report(y_true=np.argmax(y_test, axis=1),
                            y_pred=np.argmax(model2.predict(X_test), axis=1),
                            target_names=y.columns.values))

vf1_arr = history2.history["val_f1_score"]
f1_arr = history2.history["f1_score"]
plt.plot(range(len(vf1_arr)),vf1_arr)
plt.plot(range(len(f1_arr)),f1_arr)

df_res = pd.DataFrame(np.argmax(model2.predict(X_test), axis=1))

df_res = df_res.map(lambda x:y.columns.values[x])

print(classification_report(y_true=np.argmax(y_train, axis=1),
                            y_pred=np.argmax(model2.predict(X_train), axis=1),
                            target_names=y.columns.values))