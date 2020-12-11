import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def autoencoder_network(train, test, labels):

    # set encoding dim to 14 for real world data, batch size to 32, epochs to 100
    # set encoding dim to 10 for sim data, batch size to 250, epochs to 20
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
                        epochs=100,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(test, test),
                        verbose=1)
    predictions = model.predict(test)
    mean_squared_error = np.mean(np.power(test - predictions, 2), axis=1)
    mean_squared_error_non_fraud = []
    mean_squared_error_fraud = []

    for i, l in enumerate(labels):
        if l == 0:
            mean_squared_error_non_fraud.append(mean_squared_error[i])
        else:
            mean_squared_error_fraud.append(mean_squared_error[i])

    mean_squared_error_fraud = np.array(mean_squared_error_fraud)
    mean_squared_error_non_fraud = np.array(mean_squared_error_non_fraud)

    for i in range(0, 100, 10):
        print("Non fraud percentile " + str(i) + ": " + str(np.percentile(mean_squared_error_non_fraud, i)))
        print("Fraud percentile " + str(i) + ": " + str(np.percentile(mean_squared_error_fraud, i)))

    # use a threshold of 0.05 for sim data
    # use a threshold of 3 for real world data
    threshold = 3
    y_pred = [1 if e > threshold else 0 for e in mean_squared_error]
    confusion = confusion_matrix(labels, y_pred)




    data_labels = ["Valid", "Fraudulent"]

    plt.figure(figsize=(12, 12))
    sns.heatmap(confusion, xticklabels=data_labels, yticklabels=data_labels, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('Labels')
    plt.xlabel('Predictions')
    plt.show()

def lstm_network(train_inputs, train_labels, test_inputs, test_labels):
    model = tf.keras.Sequential()
    model.add(LSTM(20, input_shape=train_inputs.shape[1:], activation='relu', dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_inputs, train_labels, verbose=1, epochs=100, batch_size=10000, class_weight={0 : 1., 1: float(int(1/(np.mean(train_labels) + 0.000000001)))}, validation_split=0.3)

    predictions = model.predict(test_inputs)

    fpr, tpr, thresholds = roc_curve(train_labels, model.predict(train_inputs), pos_label=1)
    print('TRAIN | AUC Score: ' + str((auc(fpr, tpr))))
    fpr, tpr, thresholds = roc_curve(test_labels, predictions, pos_label=1)
    print('TEST | AUC Score: ' + str((auc(fpr, tpr))))

    plt.scatter(np.squeeze(predictions), np.array(test_labels))
    plt.show()
    plt.clf()

    for i in range(40, 50):
        print(i)
        y_pred = [1 if e > (i / 100) else 0 for e in np.squeeze(predictions)]
        confusion = confusion_matrix(test_labels, y_pred)

        data_labels = ["Valid", "Fraudulent"]

        plt.figure(figsize=(12, 12))
        sns.heatmap(confusion, xticklabels=data_labels, yticklabels=data_labels, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('Labels')
        plt.xlabel('Predictions')
        plt.show()