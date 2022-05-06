import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

def f1(y_true, y_pred):
    re = tf.argmax(y_pred, axis=1)
    re = tf.reshape(re, [-1,1])
    score = f1_score(y_true=y_true.numpy(), y_pred=re, average="macro")
    return score

def load_model(filepath):
    model = tf.keras.models.load_model(filepath, custom_objects={ 'f1': f1 })
    return model
