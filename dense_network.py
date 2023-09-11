import numpy as np
import tensorflow as tf
keras = tf.keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def one_hot(y):  # works only with label encoded data
    Y = np.zeros(shape=[y.shape[0], np.unique(y).shape[0]])
    for idx, value in enumerate(y):
        Y[idx, int(value)] = 1
    return Y


data = pd.read_csv('data/iris.csv')
X_, y_, = data.iloc[:, 0:-1].to_numpy(), data.iloc[:, -1].to_numpy()
y_ = LabelEncoder().fit_transform(y_)
Y_ = one_hot(y_)


def build_nn_classifier(nn_shape: np.array, in_shape, out_shape, alpha=1/1_000):
    input_layer = keras.layers.Input(shape=(in_shape, ))
    # input_layer = keras.layers.Input(shape=(None, in_shape))
    x = input_layer
    for h in nn_shape:
        x = keras.layers.Dense(units=h, activation='relu')(x)
    output_layer = keras.layers.Dense(units=out_shape, activation='sigmoid')(x)
    model = keras.models.Model(input_layer, output_layer)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=alpha),
        loss=keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )
    return model


def build_nn_regressor(nn_shape: np.array, in_shape, alpha=1/1_000):
    input_layer = keras.layers.Input(shape=(in_shape, ))
    # input_layer = keras.layers.Input(shape=(None, in_shape))
    x = input_layer
    for h in nn_shape:
        x = keras.layers.Dense(units=h, activation='relu')(x)
    output_layer = keras.layers.Dense(units=1, activation='linear')(x)
    model = keras.models.Model(input_layer, output_layer)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=alpha),
        loss=keras.losses.MeanSquaredError,
        metrics=['accuracy']
    )
    return model


X_train, X_test, y_train, y_test = train_test_split(
    X_, Y_, train_size=1/2, shuffle=True
)
nn_shape_ = np.array([16, 16, 32, 32])
m = build_nn_classifier(
    nn_shape_, in_shape=X_.shape[1], out_shape=Y_.shape[1]
)
# print(m.summary())
# m.fit(X_train, y_train, epochs=200)


















































































































































































































































































































