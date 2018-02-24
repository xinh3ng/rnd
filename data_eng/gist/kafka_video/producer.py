#|/usr/bin/env python
"""

https://scotch.io/amp/tutorials/build-a-distributed-streaming-system-with-apache-kafka-and-python

PROCEDURE:
  $ brew services stop kafka
  $ zkServer start; kafka-server-start /usr/local/etc/kafka/server.properties  # It will block the terminal
  $ brew services restart kafka
"""
import os
import time
import cv2
from kafka import SimpleProducer, SimpleClient
from pydsutils.generic import create_logger

logger = create_logger(__name__, level='info')


def video_emitter(video_file, topic, producer_port=9092):

    # Open the video
    assert os.path.isfile(video_file), 'Video does not exist'

    # Create a producer
    kafka = SimpleClient('localhost:%d' % producer_port)
    producer = SimpleProducer(kafka)
    logger.info('Kafka procuder created')

    #
    video = cv2.VideoCapture(video_file)
    logger.info('Emitting...')
    cnt = 1  # Count the frames
    while video.isOpened:
        success, image = video.read()
        if not success:  # check if the file has read to the end
            break
        ret, jpeg = cv2.imencode('.png', image)
        logger.info('Successfully read one video frame as png. Frame count = %d' % cnt)

        # Convert the image to bytes and send to kafka
        producer.send_messages(topic, jpeg.tobytes())
        time.sleep(0.2)  # To reduce CPU usage
        logger.info('Successfully send the video frame into the producer.')
        cnt += 1
    video.release()
    logger.info('Finished with emitting')


if __name__ == '__main__':
    video_emitter(video_file='./big_buck_bunny.mp4',
                  topic='kafka_bunny_video',
                  producer_port=9092)
