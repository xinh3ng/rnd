#|/usr/bin/env python
"""

https://scotch.io/amp/tutorials/build-a-distributed-streaming-system-with-apache-kafka-and-python

"""
from flask import Flask, Response
from kafka import KafkaConsumer
from pydsutils.generic import create_logger

logger = create_logger(__name__, level='info')

producer_port = 9092
topic = 'kafka_bunny_video'

# Connect to Kafka server and pass the topic we want to consume
consumer = KafkaConsumer(topic, group_id='view',
                         bootstrap_servers=['0.0.0.0:%d' % producer_port])
app = Flask(__name__)

@app.route('/')
def index():
    # return a multipart response
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def kafkastream():
    for msg in consumer:
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + msg.value + b'\r\n\r\n')


if __name__ == '__main__':
    server = 5000
    logger.info('Server name is %d' %server)
    app.run(host='0.0.0.0', port=server, debug=True)
