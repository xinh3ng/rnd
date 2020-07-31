# |/usr/bin/env python
"""
Link: https://towardsdatascience.com/elasticsearch-for-data-science-just-got-way-easier-95912d724636
"""
# Importing Eland and low-level Elasticsearch clients for comparison
import eland as ed
from eland.conftest import *
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

import json
import numpy as np
import pandas as pd


"""
Common data science use cases such as reading an entire Elasticsearch index into a pandas dataframe for Exploratory Data Analysis or training an ML model would usually require some not-so-efficient shortcuts.
"""

index_name = "kibana_sample_data_ecommerce"

es = Elasticsearch()
search = Search(using=es, index=index_name).query("match_all")  # define the search statement

documents = [hit.to_dict() for hit in search.scan()]  # Retrieve the documents from the search

# df_ecommerce = pd.DataFrame.from_records(documents)
# df_ecommerce.head()["geoip"]