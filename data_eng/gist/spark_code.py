#|/usr/bin/env python
"""

https://scotch.io/amp/tutorials/build-a-distributed-streaming-system-with-apache-kafka-and-python

$ export PYSPARK_PYTHON=`which python`; $SPARK_HOME/bin/spark-submit --master=local --packages=org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 spark_code.py
"""
import os
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import json
from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")
os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 pyspark-shell"

sc = SparkContext(appName="spark-streaming-Kafka")
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
