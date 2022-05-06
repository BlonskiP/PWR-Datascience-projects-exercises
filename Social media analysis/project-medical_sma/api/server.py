#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from model.embedder import Embedder
from model.classifier import Classifier

import sys
from flask import Flask, request

app = Flask(__name__)

embedder = None
classifier = None
label_mapping = ["Medyczne", "Medyczne-Diagnoza", "Niemedyczny"]

@app.route("/classify", methods=["POST"])
def classify_text():
    text = request.get_json()["text"]
    embeddings = embedder.make_embeddings([text])
    predictions = classifier.predict(embeddings)
    label_indices = np.argmax(predictions, axis=1)
    label_names = [label_mapping[index] for index in label_indices]
    return { "prediction": [int(i) for i in label_indices], "prediction_name": list(label_names) }
    
if __name__=='__main__':
    print(sys.argv)
    if len(sys.argv) < 3:
        print(f"Usage: flask {sys.argv[0]} <embedding-model> <classifier-model>")
        exit(1)

    embedding_model_path = sys.argv[1]
    classifier_model_path = sys.argv[2]

    # force cpu use
    cpu_devices = tf.config.list_physical_devices(device_type='CPU')
    tf.config.set_visible_devices(devices=cpu_devices, device_type='CPU')
    tf.config.set_visible_devices([], 'GPU')

    embedder = Embedder(embedding_model_path, cuda=False, batch=1, word_embeddings=True)
    classifier = Classifier(classifier_model_path)

    app.run(host='0.0.0.0', port='8000', debug=True)
