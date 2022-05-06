import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GRU
from tensorflow.keras.layers import Bidirectional

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os

from model.utils import f1

class ModelSet:

    def __init__(self, X, y, label_mappings):
        self.models = {}
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.X = X
        self.y = y
        self.label_mappings = label_mappings
        self.n_categories = len(self.label_mappings)
        # print(f"n_categories={n_categories}")
        self.input_shape = self.X.shape[-2:]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def extend_train_set(self, X_l, y_l, size=1000):
      self.X_l = X_l[:size]
      self.y_l = y_l[:size]
      self.X_train = np.concatenate((self.X_train, self.X_l))
      self.y_train = np.concatenate((self.y_train, self.y_l))

      p = np.random.permutation(len(self.y_train))
      self.X_train = self.X_train[p]
      self.y_train = self.y_train[p]

      print(self.X_train.shape, self.y_train.shape)
      for i in range(len(self.label_mappings)):
        print(self.label_mappings[i], sum(y == i))

    def name_from_layers(self, layers, units):
        name_parts=[]
        for idx, layer in enumerate(layers):
            if layer == LSTM:
                name_parts.append("LSTM")
            if layer == GRU:
                name_parts.append("GRU")
        name_parts.append(str(units))
        name = "-".join(name_parts)
        # print(f"Model {name}")
        return name

    def build_model(self, layers=[LSTM, GRU], units=64):

        name = self.name_from_layers(layers, units)

        model = Sequential(name=name)

        model.add(Input(shape=self.input_shape))

        for idx, layer in enumerate(layers):
            return_sequences = idx != len(layers) - 1
            model.add(Bidirectional(layer(units, dropout=0.2, recurrent_dropout=0, return_sequences=return_sequences)))

        model.add(Dense(self.n_categories, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', f1]
                      , run_eagerly=True #for tensor.numpy() in f1 score metric
                      )

        self.models[name] = model

        return model

    def train_model(self, model):
        model_checkpoint = self.prepare_checkpoint(model)
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                            batch_size=32, epochs=50,
                            callbacks=[self.early_stopping, model_checkpoint])

        model.load_weights(model_checkpoint.filepath)

        print("Evaluation")
        metrics = model.evaluate(self.X_test, self.y_test)
        return metrics, history

    def print_report(self, model):
        print("")
        y_test_pred = np.argmax(model.predict(self.X_test), axis=-1)
        print(classification_report(self.y_test, y_test_pred, target_names=list(self.label_mappings)))

    def build_and_train(self, params):
        for p in params:
            model = self.build_model(layers=p['layers'], units=p['units'])
            model.summary()
            self.train_model(model)
            self.print_report(model)

    def collect_results(self):
        name_a = []
        acc_a = []
        f1_a = []
        for name, model in self.models.items():
            metrics = model.evaluate(self.X_test, self.y_test)
            print(f"{name} {metrics}")
            name_a.append(name)
            acc_a.append(metrics[1])
            f1_a.append(metrics[2])

        df = pd.DataFrame({
            'name' : name_a,
            'accuracy' : acc_a,
            'f1score' : f1_a
        })

        return df

    def prepare_checkpoint(self, model):

        checkpoint_filepath = os.path.join("checkpoints",
                                           model.name.replace(" ", "-") + "_checkpoint")

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_f1',
            mode='max',
            save_best_only=True)

        return model_checkpoint
