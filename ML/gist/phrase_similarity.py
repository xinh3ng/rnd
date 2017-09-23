#|/usr/bin/env python

"""
http://outlace.com/rlpart1.html
"""
from __future__ import division
from pdb import set_trace as debug
import spacy

from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en")
logger.info("Successfully loaded the spacy model")

keyword = nlp(u"katy perry")
for d in ["worst fries", "katy perry tickets", "bruno mars tickets"]:
    doc = nlp(u"%s" % d)
    print d, " : ", keyword.similarity(doc)

logger.info("ALL DONE\n")
