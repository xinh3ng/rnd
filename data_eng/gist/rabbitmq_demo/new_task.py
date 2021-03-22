"""
This is the producer script

https://www.rabbitmq.com/tutorials/tutorial-one-python.html

https://www.rabbitmq.com/tutorials/tutorial-two-python.html

brew services start rabbitmq

"""
import pika
import sys
from typing import List

#
queue_name = "hello"
message = " ".join(sys.argv[1:]) or "Hello World!"

#
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue=queue_name)

channel.basic_publish(exchange="", routing_key=queue_name, body=message)
print(f"[x] Successfully sent '{message}'")

# Closing it
connection.close()
