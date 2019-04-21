# |/usr/bin/env python

"""

https://bitbucket.org/yunazzang/aiwiththebest_byor/src/6e80c71ddd79f96ae423684523cebd27099e0a94/PhraseSimilarity.py?at=master&fileviewer=file-view-default
https://bitbucket.org/yunazzang/aiwiththebest_byor
"""
from __future__ import division
from pdb import set_trace as debug
import os
import sys
import numpy as np
import math
from scipy.spatial import distance
from gensim.models import word2vec
from random import sample
from nltk.corpus import stopwords
from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")

# Get a pretrained word vector

# Change this to your own path.
bin_vectors_file = "%s/data/nlp/GoogleNews-vectors-negative300.bin.gz" % os.environ["HOME"]

logger.info("Loading the data file. Please wait...")
model1 = word2vec.Word2Vec.load_word2vec_format(bin_vectors_file, binary=True)
logger.info("Successfully loaded 3GB bin file")


logger.info("ALL DONE\n")
