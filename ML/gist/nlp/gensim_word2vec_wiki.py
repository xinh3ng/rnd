#|/usr/bin/env python

"""
https://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
Data: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
"""

from pdb import set_trace as debug
import os
import multiprocessing
from gensim.models import word2vec
from gensim.corpora import WikiCorpus
from gensim.models.word2vec import LineSentence

from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


def process_wiki(raw_file="enwiki-latest-pages-articles.xml.bz2",
                 processed_file="wiki.en.text"):
    """Process all wikipedia articles
    """
    cnt = 0
    output =  open(processed_file, "w")
    wiki = WikiCorpus(raw_file, lemmatize=False, dictionary={})

    for text in wiki.get_texts():  # Get text article by article
        output.write(bytes(" ".join(text), "utf-8").decode("utf-8") + '\n')
        cnt += 1
        if (cnt % 10000 == 0):
            logger.info("Saved " + str(cnt) + " articles")

    output.close()
    logger.info("Successfully saved all %d processed articles in: %s" %(cnt, processed_file))
    return

#
is_process_wiki = True
raw_wiki_file = "./enwiki-latest-pages-articles.xml.bz2"
processed_wiki_file = "/tmp/wiki.en.text"
model_file = "word2vec_model.wiki_en"

if is_process_wiki:
    process_wiki(raw_file=raw_wiki_file,
                 processed_file=processed_wiki_file)

model = word2vec.Word2Vec(LineSentence(processed_wiki_file),  size=400, window=5,
                          min_count=5, workers=(-1 + multiprocessing.cpu_count()))
logger.info("Successfully init the wor2vec model")

# trim unneeded model memory = use (much) less RAM
model.init_sims(replace=True)
model.save(model_file)


model.most_similar("queen")


logger.info("ALL DONE\n")

