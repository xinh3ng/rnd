#|/usr/bin/env python
"""

Language models
  "en", en_core_web_sm, en_core_web_md
"""
from __future__ import division
from pdb import set_trace as debug
import spacy
import en_core_web_sm

from pydsutils.generic import create_logger
logger = create_logger(__name__, level="info")

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = en_core_web_sm.load()
logger.info("Successfully loaded the spacy model")

doc_1 = nlp(u"ATT park")
doc_2 = nlp(u"AT&T Park")
print(doc_1.similarity(doc_2))
debug()

logger.info("ALL DONE\n")
