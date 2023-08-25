import time
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from preprocess import get_data


def load_data():
    """
    提取训练集和验证集
    :returns    训练集和验证集
    """
    # (2170, 1, 4, 507) RNA序列为（4，507）矩阵，共2170条数据 转换为(2170,4,507)
    train_bags, train_labels = get_data('ALKBH5_Baltz2012.train.positives.fa', 'ALKBH5_Baltz2012.train.negatives.fa',
                                        channel=1, window_size=501)
    val_bags, val_labels = get_data('ALKBH5_Baltz2012.val.positives.fa', 'ALKBH5_Baltz2012.val.negatives.fa', channel=1,
                                    window_size=501)
    return train_bags, train_labels, val_bags, val_labels


def vote():
    import numpy as np
    probs_models = []
    train_bags, train_labels, val_bags, val_labels = load_data()
    validation_data = np.array(val_bags)
    for i in range(3):
        # 加载模型
        model = tf.keras.models.load_model(f'path/to/saved/model{i}')
        # 使用模型进行预测或其他操作
        predictions = model.predict(validation_data)  # todo 假设输出结果是列表

        probs_models.append(np.argmax(predictions, axis=1).tolist())

    # 硬投票集成
    count_1 = np.sum(np.array(probs_models) == 1, axis=0)
    count_0 = np.sum(np.array(probs_models) == 0, axis=0)

    # 输出数量较多的数
    output = np.where(count_1 > count_0, 1, 0)  # todo output格式为ndarray
    # 计算output与label的相关性，从而计算auc

    # 创建 AUC 指标对象
    auc_metric = tf.keras.metrics.AUC()

    # 创建模拟的预测值和标签
    predictions = tf.constant(output)

    # 更新 AUC 指标的状态
    auc_metric.update_state(val_labels, predictions)

    # 获取 AUC 值
    auc_score = auc_metric.result()

    print("AUC:", auc_score.numpy())


def init_model0():
    """线性网络"""
    model_name = "{}-nodes-{}-dense-{}".format(32, 1,
                                               int(time.time()))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}', histogram_freq=1)
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.25, input_shape=(507, 4)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    return model, tensorboard_callback


def init_model1():
    """cnn网络"""
    model_name = "{}-nodes-{}-conv-{}-nodes-{}-dense-{}".format(64, 1,
                                                                64,
                                                                3,
                                                                int(time.time()))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}',
                                                          histogram_freq=1)
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.25, input_shape=(507, 4)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                     activation='relu'))
    model.add(tf.keras.layers.Flatten())
    for _ in range(2):
        model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    return model, tensorboard_callback


def init_model2():
    """lstm网络"""
    model_name = "{}-nodes-{}-lstm-{}-nodes-{}-dense-{}".format(128, 2,
                                                                128, 2,
                                                                int(time.time()))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}',
                                                          histogram_freq=1)
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.25, input_shape=(507, 4)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation='tanh',
                                   recurrent_activation='sigmoid', recurrent_dropout=0))
    model.add(tf.keras.layers.LSTM(units=128, activation='relu', recurrent_activation='sigmoid',
                                   recurrent_dropout=0))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    return model, tensorboard_callback


def train():
    # conv_layers = [1, 2, 3]  # , 2, 1
    # dense_layers = [1, 2, 3, 4]  # 1, 2, 3, 4
    # layer_sizes = [32, 64, 128, 256]  # 32, 64, 128, 256
    # conv_layers_sizes = [32, 64, 128, 256]  # 32, 64, 128, 256
    train_bags, train_labels, val_bags, val_labels = load_data()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    train_labels = [[1, 0] if i == 0 else [0, 1] for i in train_labels]
    val_labels = [[1, 0] if i == 0 else [0, 1] for i in val_labels]
    for model, tensorboard_callback in init_model0():
        # model.build(np.array(train_bags).shape)
        # print(model.summary())
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        # 定义学习率衰减回调函数
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5)
        model.fit(np.array(train_bags), np.array(train_labels), batch_size=32, epochs=500, verbose=1,
                  validation_data=(np.array(val_bags), np.array(val_labels)),
                  callbacks=[early_stopping, reduce_lr])
        model.save('D:/PythonProject/mlearning_homework_tf/model/model0')


if __name__ == '__main__':
    train()
