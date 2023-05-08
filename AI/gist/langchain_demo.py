"""

# 
"""
import json
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
import os
import pandas as pd


loader = TextLoader("../state_of_the_union.txt", encoding="utf8")

print("ALL DONE!\n")
