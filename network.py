import tensorflow

def seq_network(train, test):
    model = tf.keras.Sequential()
    model.add(Input(shape=(train.shape[1], )))
    model.add(Dense(encoding_dim, activation="tanh"))
    model.add(Dense(int(encoding_dim / 2), activation="relu"))
    model.add(Dense(int(encoding_dim / 2), activation='tanh'))
    model.add(Dense(train.shape[1], activation='relu'))
    model.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])
    model.fit(train, test,
                        epochs=100,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(test, test),
                        verbose=1)