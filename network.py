import tensorflow
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding

def autoencoder_network(train, test):
    model = tf.keras.Sequential()
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

def lstm_network():
    model = tf.keras.Sequential()
    # model.add(Embedding(input_dim=1000, output_dim=64))
    # model.add(LSTM(128))
    # model.add(Dense(10))
    pass

def transformer_network():
    model = tf.keras.Sequential()
    pass