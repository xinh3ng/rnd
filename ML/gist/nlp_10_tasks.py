#|/usr/bin/env python
"""
"""
import os
from pydsutils.generic import create_logger
logger = create_logger(__name__)

# 1. stemming
from stemming.porter2 import stem
logger.info("Stemming result: %s" % stem("casually"))

# 2. Lemma
# python -m spacy download en
import spacy
nlp = spacy.load("en")
doc = "good better best"
logger.info("Lemmanization")
for token in nlp(doc):
    print(token, " -> ", token.lemma_)

# 3 word2vec
import gensim
from gensim.models.keyedvectors import KeyedVectors
logger.info("word2vec, loading the model...")
word_vectors = KeyedVectors.load_word2vec_format(
        "%s/data/nlp/GoogleNews-vectors-negative300.bin.gz" % os.environ["HOME"], 
        binary=True)
logger.info("Word vec of human is: %s " % str(word_vectors["human"]))

# 用 gensim 训练你自己的词向量
sentences = [["first", "sentence"], ["second","sentence"]]
model = gensim.models.Word2Vec(sentences, min_count=1, size=300, workers=4)
logger.info("word2vec model is trained")

# 4. Tokenize
nlp = spacy.load("en")
sentence = "Ashok killed the snake with a stick"
logger.info("Tokenize")
for token in nlp(sentence):
   print(token, token.pos_)

# 5. Entity Disambiguation


# 6. Named Entity Recognition
logger.info("Named entity recognition")
nlp = spacy.load("en")
sentence = "Ram of Apple Inc. traveled to Sydney on 5th October 2017"
for token in nlp(sentence):
   print(token, token.ent_type_)

# 7. Sentiment analytis

# 8. Text similarity
 
# 9. Language recognition

# 10. Sentence summarization
from gensim.summarization import summarize
sentence = "Automatic summarization is the process of shortening a text document with software, in order to create a summary with the major points of the original document. Technologies that can make a coherent summary take into account variables such as length, writing style and syntax.Automatic data summarization is part of machine learning and data mining. The main idea of summarization is to find a subset of data which contains the information of the entire set. Such techniques are widely used in industry today. Search engines are an example; others include summarization of documents, image collections and videos. Document summarization tries to create a representative summary or abstract of the entire document, by finding the most informative sentences, while in image summarization the system finds the most representative and important (i.e. salient) images. For surveillance videos, one might want to extract the important events from the uneventful context.There are two general approaches to automatic summarization: extraction and abstraction. Extractive methods work by selecting a subset of existing words, phrases, or sentences in the original text to form the summary. In contrast, abstractive methods build an internal semantic representation and then use natural language generation techniques to create a summary that is closer to what a human might express. Such a summary might include verbal innovations. Research to date has focused primarily on extractive methods, which are appropriate for image collection summarization and video summarization."

logger.info("Sentence summarization:")
logger.info("%s" % summarize(sentence))

# 
logger.info("ALL DONE\n")
