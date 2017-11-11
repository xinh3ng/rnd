#|/usr/bin/env python
"""Visual QA system, Keras

https://github.com/iamaaditya/VQA_Demo/blob/master/Visual_Question_Answering_Demo_in_python_notebook.ipynb
"""
from pdb import set_trace as debug
import os
import numpy as np
from sklearn.externals import joblib
import cv2
import keras
keras.backend.set_image_dim_ordering('th')  # theano backend
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
import spacy
from pydsutils.generic import create_logger

from VGG import VGG_16

logger = create_logger(__name__)


def get_image_model(cnn_weights_file):
    """Read the CNN weights file return the VGG model update with the weights.
    
    """
    image_model = VGG_16(cnn_weights_file)

    # this is standard VGG 16 without the last two layers
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # one may try 'adam', but the loss function for this kind of task is pretty standard
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model


def get_image_features(image_file, cnn_weights_file):
    """Run the image_file to VGG 16 model and return the weights (filters) as (1, 4096) vector
    
    """
    # Magic_Number = 4096  > Comes from last layer of VGG Model
    image_features = np.zeros((1, 4096))
    
    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file), (224, 224))
    im = im.transpose((2, 0, 1)) # convert the image to RGBA

    # this axis dimension is required because VGG was trained on a dimension of (1, 3, 224, 224) (first axis 
    # is for the batch size even if we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)
    image_features[0, :] = get_image_model(cnn_weights_file).predict(im)[0]
    return image_features


def get_question_features(question):
    """For a given question, a unicode string, returns the time series vector
    with each word (token) transformed into a (300,)  representation
    calculated using Glove Vector
    
    """
    # Usea the given glove model
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, len(tokens), 300))
    for j in range(len(tokens)):
        question_tensor[0, j, :] = tokens[j].vector[0:300]  # force to be 300 elements
    return question_tensor


def get_vqa_model(vqa_model_file, vqa_weights_file):
    ''' Given the VQA model and its weights, compiles and returns the model '''

    # thanks the keras function for loading a model from JSON, this becomes
    # very easy to understand and work. Alternative would be to load model
    # from binary like cPickle but then model would be obfuscated to users
    vqa_model = model_from_json(open(vqa_model_file).read())
    vqa_model.load_weights(vqa_weights_file)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model


#################################
#################################

vqa_model_file = '%s/data/nlp/VQA/VQA_MODEL.json' % os.environ['HOME']
vqa_weights_file = '%s/data/nlp/VQA/VQA_MODEL_WEIGHTS.hdf5' % os.environ['HOME']
label_encoder_file = '%s/data/nlp/VQA/FULL_labelencoder_trainval.pkl' % os.environ['HOME']
cnn_weights_file = '%s/data/nlp/VQA/vgg16_weights.h5' % os.environ['HOME']

# 
model_vgg = get_image_model(cnn_weights_file)
plot_model(model_vgg, to_file='model_vgg.png')

# Test the word embeddings
word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
obama = word_embeddings(u"obama")
putin = word_embeddings(u"putin")
banana = word_embeddings(u"banana")
monkey = word_embeddings(u"monkey")
logger.info('obama-putin similarity: %.4f' % obama.similarity(putin))
logger.info('obama-banana similarity: %.4f' % obama.similarity(banana))
logger.info('banana-monkey similarity: %.4f' % banana.similarity(monkey))

# 
model_vqa = get_vqa_model(vqa_model_file, vqa_weights_file)
plot_model(model_vqa, to_file='model_vqa.png')

#########################
# VQA demo example
#########################
image_file = '%s/data/nlp/VQA/test.jpg' % os.environ['HOME']
question = u"What vehicle is in the picture?"

image_features = get_image_features(image_file, cnn_weights_file)  # get the image features

question_features = get_question_features(question)  # get the question features

y_output = model_vqa.predict([question_features, image_features])

# This task here is represented as a classification into a 1000 top answers
# this means some of the answers were not part of training and thus would not show up in the result.
labelencoder = joblib.load(label_encoder_file)
for label in reversed(np.argsort(y_output)[0, -5:]):
    logger.info("%s" % str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))

# 
logger.info('ALL DONE\n')
