"""
https://www.rabbitmq.com/tutorials/tutorial-one-python.html

brew services start rabbitmq

"""
import pika
from typing import List

#
queue_name = "hello"
mesg = "Hello World!"

#
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue=queue_name)

channel.basic_publish(exchange="", routing_key=queue_name, body=mesg)
print(f"[x] Sent '{mesg}'")

# Closing it
connection.close()
