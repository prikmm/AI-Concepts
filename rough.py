import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from optuna import create_study, study
import numpy as np

def data():
    mnist = keras.datasets.mnist.load_data()
    train = mnist[0]
    test = mnist[1]
    X_train = train[0]
    y_train = train[1]
    X_test = test[0]
    y_test = test[1]
    return (X_train, y_train, X_test, y_test)

class ExponentialLearningRate(keras.callbacks.Callback):

    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        K = keras.backend
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def get_callbacks(factor):
    callbacks = [keras.callbacks.EarlyStopping(patience=10),
                 ExponentialLearningRate(factor=factor),]
    return callbacks

def nn_clf(trial):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    optimizer = keras.optimizers.Nadam(lr=2e-4)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
                        epochs=1,
                        validation_data=(X_valid, y_valid),
                        callbacks = get_callbacks(trial.suggest_categorical('factor', [1.005, 1.01])))
    
    loss = np.min(history.history['val_loss'])
    return loss


if __name__ == '__main__':
    
    X_train, y_train, X_test, y_test = data()
    for name, data, target in (("Train", X_train, y_train), ("Test", X_test, y_test)):
        print(f'{name}:----')
        print("Data: ", data.shape)
        print("Target: ", target.shape)
        print("\n")

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    for name, data, target in (("Train", X_train, y_train), ("Valid", X_valid, y_valid)):
        print(f'{name}:----')
        print("Data: ", data.shape)
        print("Target: ", target.shape)
        print("\n")

    X_train = X_train.astype('float32') / 255
    X_valid = X_valid.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    study = create_study()
    study.optimize(nn_clf, n_trials=10)
    print(study.best_params)
