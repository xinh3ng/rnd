#|/usr/bin/env python
"""

"""
import os
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import json
from pydsutils.generic import create_logger

logger = create_logger(__name__, level='info')
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-8:2.2.1 pyspark-shell'

sc = SparkContext(appName='spark streaming kafka')
sc.setLogLevel("ERROR")
ssc = StreamingContext(sc, 60)

# Zookeeper quorum
# group id
# topics
stream = KafkaUtils.createStream(ssc, 'cdh57-01-node-01.moffatt.me:2181', "spark-streaming", {"twitter":1})
parsed = stream.map(lambda v: json.loads(v[1]))
parsed.count().map(lambda x:'Tweets in this batch: %s' % x).pprint()

authors_dstream = parsed.map(lambda tweet: tweet["user"]["screen_name"])
author_counts = authors_dstream.countByValue()
author_counts.pprint()

logger.info("ALL DONE!\n")

