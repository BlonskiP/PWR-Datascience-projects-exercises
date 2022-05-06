#!/usr/bin/env python3
#
# Make embedding using allegro herbert
#

import sys
from csv import DictReader
import argparse
import numpy as np
from embedder import Embedder, HERBERT

parser = argparse.ArgumentParser(description="Make embeddings using allegro helbert")
parser.add_argument('inputcsv', help="Input csv file with columns text,label")
parser.add_argument('outputfile', help="Path to output file")
parser.add_argument('--model', type=str, default=HERBERT, help="Bert model name or path to custom one")
parser.add_argument('--batch', type=int, default=32, help="GPU batch size")
parser.add_argument('--word', action="store_true", help="Output word embeddings instead of sentence")
args = parser.parse_args()

input_fp = args.inputcsv
output_fp = args.outputfile
batch = args.batch
word_mode = args.word
model_type = args.model

data = { "text": [], "label": [] }
with open(input_fp, 'r') as f:
    reader = DictReader(f)
    for row in reader:
        data["text"].append(row["text"])
        data["label"].append(row["label"])

embedder = Embedder(model=model_type, batch=batch, word_embeddings=word_mode)

# tokenize
texts = data['text']
all_sentence_embeddings = embedder.make_embeddings(texts)

# labels
labels = np.array(data['label'])
label_indices = np.zeros(labels.shape, dtype=np.int64)
unique = np.unique(labels)
for i, label in enumerate(unique):
    label_indices[labels == label] = i

# save to file
np.savez_compressed(output_fp, embeddings=all_sentence_embeddings, labels=label_indices, label_mappings=unique)
