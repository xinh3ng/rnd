#|/usr/bin/env python
"""

https://scotch.io/amp/tutorials/build-a-distributed-streaming-system-with-apache-kafka-and-python

$ zkServer start; kafka-server-start /usr/local/etc/kafka/server.properties
$ brew services start kafka
"""
import os
import time
import cv2
from kafka import SimpleProducer, KafkaClient
from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")

port = 9092
topic = "kafka_video"

#  connect to Kafka
kafka = KafkaClient("localhost:%d" %port)
producer = SimpleProducer(kafka)


def video_emitter(video_file):

    # Open the video
    assert os.path.isfile(video_file), "Video does not exist"
    video = cv2.VideoCapture(video_file)
    logger.info("Emitting...")

    while video.isOpened:
        success, image = video.read()
        if not success:  # check if the file has read to the end
            break
        ret, jpeg = cv2.imencode(".png", image)

        # Convert the image to bytes and send to kafka
        producer.send_messages(topic, jpeg.tobytes())

        time.sleep(0.2)  # To reduce CPU usage create sleep time of 0.2sec

    video.release()
    logger.info("Finished with emitting")


if __name__ == "__main__":
    video_emitter("./big_buck_bunny.mp4")
