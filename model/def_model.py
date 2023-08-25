"""{64}-nodes-{1}-conv-{64}-nodes-{3}-dense
def init_model(dense_layers, layer_sizes, conv_layers):
    models = []
    tensorboard_callbacks = []
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                model_name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer,
                                                                   int(time.time()))
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}', histogram_freq=1)
                tensorboard_callbacks.append(tensorboard_callback)
                model = Sequential()
                model.add(
                    tf.keras.layers.Conv1D(filters=32, data_format='channels_first', kernel_size=3, activation='relu'))
                model.add(
                    tf.keras.layers.MaxPool1D(pool_size=2))
                for i in range(conv_layer - 1):
                    model.add(tf.keras.layers.Conv1D(filters=64, data_format='channels_first', kernel_size=3,
                                                     activation='relu'))
                    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
                model.add(tf.keras.layers.Flatten())
                for i in range(dense_layer):
                    model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
                model.add(tf.keras.layers.Dense(2, activation='softmax'))
                models.append(model)
    return models, tensorboard_callbacks
"""

""" best {32}-nodes-{1}-dense 1 dense is important,nodes is not
def init_model(dense_layers, layer_sizes):
    models = []
    tensorboard_callbacks = []
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:

            model_name = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer,
                                                       int(time.time()))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}', histogram_freq=1)
            tensorboard_callbacks.append(tensorboard_callback)
            model = Sequential()
            model.add(tf.keras.layers.Flatten())
            for _ in range(dense_layer):
                model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            model.add(tf.keras.layers.Dense(2, activation='softmax'))
            models.append(model)
    return models, tensorboard_callbacks
"""
""" {128}-nodes-{2}-lstm-{128}-nodes-{2}-dense
def init_model(dense_layers, layer_sizes, lstm_layers):
    models = []
tensorboard_callbacks = []
    for lstm_layer in lstm_layers:
        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                model_name = "{}-lstm-{}-nodes-{}-dense-{}".format(lstm_layer, layer_size, dense_layer,
                                                                   int(time.time()))
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}', histogram_freq=1)
                tensorboard_callbacks.append(tensorboard_callback)
                model = Sequential()
                for _ in range(lstm_layer-1):
                    model.add(tf.keras.layers.LSTM(layer_size, return_sequences=True))
                model.add(tf.keras.layers.LSTM(units=layer_size))
                for _ in range(dense_layer):
                    model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
                model.add(tf.keras.layers.Dense(2, activation='softmax'))
                models.append(model)
    return models, tensorboard_callbacks

"""