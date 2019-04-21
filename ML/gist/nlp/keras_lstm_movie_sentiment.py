# |/usr/bin/env python
"""Sequence classification, Movie sentiment, Keras, LSTM

https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
"""
from pdb import set_trace as debug
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")
SEED = 7

##############################
top_words = 5000
max_review_length = 500
embedding_vector_length = 32
epochs = 10

np.random.seed(SEED)

(X_train, y_train), (X_val, y_val) = imdb.load_data(num_words=top_words)

# Truncate or pad input sequences to a given length
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_val = sequence.pad_sequences(X_val, maxlen=max_review_length)

# Gen the model
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))  # word embedding
model.add(Dropout(0.2))  # dropout to reduce overfitting
model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))  # 100 nodes
model.add(Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]  # this is binary classification problem
)
logger.info("Model summary:%s" % model.summary())

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=64)
logger.info("Finished with model fitting")

# Final evaluation of the model
score, acc = model.evaluate(X_val, y_val, verbose=0)
logger.info("Test score:    %.4f" % (score))
logger.info("Test accuracy: %.4f" % (acc))

logger.info("ALL DONE!\n")
