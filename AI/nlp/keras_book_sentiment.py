# |/usr/bin/env python
"""Movie sentiment, Keras, LSTM

https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-using-word-embeddings.ipynb

TODO xheng: still dysfunctional because model data for a normal NN and a RNN model is different
"""
from pdb import set_trace as debug
import os
from collections import OrderedDict
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN

from pydsutils.generic import create_logger

plt.style.use("ggplot")
logger = create_logger(__name__, level="info")

# Parameters
imdb_dir = "{}/data/aclImdb".format(os.environ["HOME"])
max_words = 10000  # Only consider the top words in the data set, max_words is max_features
maxlen = 100  # Only the first100 words
train_samples = 24000  # Train sample size
val_samples = 1000  # Val sample size

glove_dir = "{}/data/glove-wikipedia".format(os.environ["HOME"])
embedding_dim = 100
epochs = 10  #
batch_size = 32
loss = "binary_crossentropy"
metrics = ["acc"]
show_plot = False
model_wts_file = "pre_trained_glove_model.h5"


def load_model_data(dir):
    texts, labels = [], []
    for label_type in ["neg", "pos"]:
        dir_name = os.path.join(dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == ".txt":
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == "neg":
                    labels.append(0)  # 0 meeas negative
                else:
                    labels.append(1)
    return texts, labels


def gen_fn_model(embedding_matrix, optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"]):
    """Generate a forward net model"""
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Load the GloVe embeddings in the model¶
    # The 1st Embedding layer has a single weight matrix: a 2D float matrix where each entry i is the word vector
    #   meant to be associated with index i
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False  # Freeze the embedding layer
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("gen_fn_model() successfully created")
    return model


def gen_simple_rnn_model(embedding_matrix, optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"]):
    model = Sequential()
    model.add(Embedding(max_words, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))  # This last layer only returns the last outputs

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("gen_fn_model() successfully created")
    return model


#######################################################################################

# labels: Sentiment scores of the text
# texts: A list of texts
texts, labels = load_model_data(dir=os.path.join(imdb_dir, "train"))

logger.info("Started to tokenize texts")
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
logger.info("Found %d unique tokens." % len(tokenizer.word_index))

logger.info("Started to pad each sequence to the same length")
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
logger.info("Shape of data tensor: %s" % str(data.shape))
logger.info("Shape of label tensor: %s" % str(labels.shape))

# Split the data into train and val set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]  # Shuffle the data, since sample were ordered: negative first, then all positive
labels = labels[indices]

x_train, y_train = data[:train_samples], labels[:train_samples]
x_val = data[train_samples : train_samples + val_samples]
y_val = labels[train_samples : train_samples + val_samples]
logger.info("Shape of x_train is: %s" % str(x_train.shape))
logger.info("Shape of x_val is: %s" % str(x_val.shape))

#
embeddings_index = {}
for line in open(os.path.join(glove_dir, "glove.6B.100d.txt")):
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype="float32")
logger.info("Found %d word vectors" % len(embeddings_index))

embedding_matrix = np.zeros((max_words, embedding_dim))  # Words not found in embedding index default to 0.
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i >= max_words:
        continue
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
logger.info("Showing the last embedding_matrix sample: index=%d, word=%s" % (i, word))

texts, labels = load_model_data(dir=os.path.join(imdb_dir, "test"))
sequences = tokenizer.texts_to_sequences(texts)  # Use the same tokenizer
x_test, y_test = pad_sequences(sequences, maxlen=maxlen), np.asarray(labels)

#####################################
# Define a model and train it
#####################################
# Define a model
models = OrderedDict(
    [
        ("fwd", gen_fn_model(embedding_matrix, optimizer="rmsprop", loss=loss, metrics=metrics)),
        ("simple-rnn", gen_simple_rnn_model(embedding_matrix, optimizer="rmsprop", loss=loss, metrics=metrics)),
    ]
)
for name, m in models.items():
    logger.info("Started to train model: %s..." % name)
    logger.info("Showing model summary:\n%s" % str(m.summary()))
    history = m.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    m.save_weights(model_wts_file)

    logger.info("Preparing plots on the model training history...")
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111)
    # ax.plot(range(1, epochs + 1), acc, 'bo', label='Training acc')
    # ax.plot(range(1, epochs + 1), val_acc, 'b', label='Validation acc')
    # ax.legend()
    # ax.plot(range(1, epochs + 1), loss, 'bo', label='Training loss')
    # ax.plot(range(1, epochs + 1), val_loss, 'b', label='Validation loss')
    # ax.legend()

    if show_plot:
        plt.show()
    else:  # Save the plot as png
        pass

    #####################################
    # Apply model on test data
    #####################################
    m.load_weights(model_wts_file)
    test_res = m.evaluate(x_test, y_test)
    logger.info("Performance on test data: %s. First value is loss" % str(test_res))
    logger.info("Successfully trained model: %s\n" % name)

#
logger.info("ALL DONE!\n")
