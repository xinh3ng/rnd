#|/usr/bin/env python
"""
"""
import os
import numpy as np
import pandas as pd
import spacy


from pydsutils.generic import create_logger

logger = create_logger(__name__)

nlp = spacy.load('en')

print('Sanity test:')
doc = nlp(u'I am a good programmer!')
print([t for t in doc])

#
logger.info('ALL DONE\n')
