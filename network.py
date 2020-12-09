import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from sklearn.metrics import confusion_matrix, roc_curve, auc
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
                        epochs=10,
                        batch_size=250,
                        shuffle=True,
                        validation_data=(test, test),
                        verbose=1)


    # Based on paper, we used a confusion matrix to visualize accuracy
    predictions = model.predict(test)
    #print(predictions)
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

    threshold = np.percentile(mean_squared_error_fraud, 25)
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
    model.fit(train_inputs, train_labels, epochs=100, batch_size=10000, class_weight={0 : 1., 1: float(int(1/np.mean(train_labels)))}, validation_split=0.3)

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

    #mean_squared_error = np.mean(np.power(test_labels - np.squeeze(predictions), 2))

    #y_pred = [1 if e > 3 else 0 for e in mean_squared_error]


def transformer_network():
    model = tf.keras.Sequential()
    pass


'''X = pd.read_csv('data/creditcard.csv', na_filter=True)

y_original = np.array(X['Class'], dtype='float')

X.drop(['Class'], inplace=True, axis=1)

rolling_window_size = 10  ### this selects how many historical transactions should be analyzed to judge the transaction at hand -- RNN width

X_interim = np.zeros([(X.shape[0]-rolling_window_size)*10,30])
y = []
for i in range((X.shape[0]-rolling_window_size)):
    beg = 0+i
    end = beg+rolling_window_size
    s = np.array(X[beg:end], dtype='float')
    X_interim[(rolling_window_size*i):(rolling_window_size*(i+1)),:] = s
    y.append(y_original[end])
 
y = np.array(y, dtype='float')
X_interim = X_interim[:,1::]
X_tensor = X_interim.reshape((int(np.shape(X_interim)[0]/rolling_window_size)), rolling_window_size, np.shape(X_interim)[1])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

test_train_split = 0.5
stratify = True

if stratify:
    y = np.vstack((range(len(y)),y)).T
    y_pos = y[y[:,1]==1]
    y_neg = y[y[:,1]==0]
    
    y_pos = y_pos[np.random.choice(y_pos.shape[0], int(y_pos.shape[0]*test_train_split), replace=False),:]
    y_neg = y_neg[np.random.choice(y_neg.shape[0], int(y_neg.shape[0]*test_train_split), replace=False),:]
    
    train_idx = np.array(np.hstack((y_pos[:,0],y_neg[:,0])), dtype='int')
    
    X_train = X_tensor[train_idx, :, :]
    X_test = np.delete(X_tensor, train_idx, axis=0)
    y_train = y[train_idx,1]
    y_test = np.delete(y, train_idx, axis=0)
    y_test = y_test[:,1]
else: 
    train_idx = np.random.choice(X_tensor.shape[0], int(X_tensor.shape[0]*test_train_split), replace=False)
    X_train = X_tensor[train_idx, :, :]
    X_test = np.delete(X_tensor, train_idx, axis=0)
    y_train = y[train_idx]
    y_test = np.delete(y, train_idx, axis=0)

del (X_tensor, y, stratify, test_train_split, train_idx, y_neg, y_pos)


### Hyperparameters Tuning
# First test optimal epochs holding everything else constant
# Dropout: 0.1-0.6
# GradientClipping: 0.1-10
# BatchSize: 32,64,128,256,512 (power of 2)


### Train LSTM using Keras 2 API ###'''
'''model = Sequential()
model.add(LSTM(20, input_shape=X_train.shape[1:], kernel_initializer='lecun_uniform', activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.1), recurrent_regularizer=tf.keras.regularizers.l1(0.01), bias_regularizer=None, activity_regularizer=None, dropout=0.2, recurrent_dropout=0.2))#, return_sequences=True))
#model.add(LSTM(12, activation='relu', return_sequences=True))
#model.add(LSTM(8, activation='relu'))
model.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #optimizer='rmsprop'
print(model.summary())

model.fit(X_train, y_train, epochs=200, batch_size=10000, class_weight={0 : 1., 1: float(int(1/np.mean(y_train)))}, validation_split=0.3)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

### test AUC ###
from sklearn import metrics 

fpr, tpr, thresholds = metrics.roc_curve(y_train, train_predict, pos_label=1)
print('TRAIN | AUC Score: ' + str((metrics.auc(fpr, tpr))))
fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predict, pos_label=1)
print('TEST | AUC Score: ' + str((metrics.auc(fpr, tpr))))

print(test_predict)

for i in range(40, 60):
    y_pred = [1 if e > (i / 100) else 0 for e in test_predict]
    confusion = confusion_matrix(y_test, y_pred)

    data_labels = ["Valid", "Fraudulent"]

    plt.figure(figsize=(12, 12))
    sns.heatmap(confusion, xticklabels=data_labels, yticklabels=data_labels, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('Labels')
    plt.xlabel('Predictions')
    plt.show()'''


