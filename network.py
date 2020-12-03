import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def autoencoder_network(train, test, labels):
    model = tf.keras.Sequential()
    encoding_dim = 14
    model.add(Input(shape=(train.shape[1], )))
    model.add(Dense(encoding_dim, activation="tanh"))
    model.add(Dense(int(encoding_dim / 2), activation="relu"))
    model.add(Dense(int(encoding_dim / 2), activation='tanh'))
    model.add(Dense(train.shape[1], activation='relu'))
    model.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])
    model.fit(train, train,
                        epochs=1,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(test, test),
                        verbose=1)


    # Based on paper, we used a confusion matrix to visualize accuracy
    predictions = model.predict(test)
    mean_squared_error = np.mean(np.power(test - predictions, 2), axis=1)

    y_pred = [1 if e > 3 else 0 for e in mean_squared_error]
    confusion = confusion_matrix(labels, y_pred)

    data_labels = ["Valid", "Fraudulent"]

    plt.figure(figsize=(12, 12))
    sns.heatmap(confusion, xticklabels=data_labels, yticklabels=data_labels, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('Labels')
    plt.xlabel('Predictions')
    plt.show()

def lstm_network(train_inputs, train_labels):
    model = Sequential()
    model.add(LSTM(20, input_shape=X_train.shape[1:], activation='relu', dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_inputs, train_labels, epochs=200, batch_size=10000, class_weight={0 : 1., 1: float(int(1/np.mean(train_labels)))}, validation_split=0.3)

def transformer_network():
    model = tf.keras.Sequential()
    pass


