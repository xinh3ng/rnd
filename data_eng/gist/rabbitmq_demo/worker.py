"""
This is the receiver script

https://www.rabbitmq.com/tutorials/tutorial-one-python.html
https://www.rabbitmq.com/tutorials/tutorial-two-python.html

"""
import pika
import time


def callback(ch, method, properties, body):
    """Standard callback to receive messages from a queue"""
    print("[x] callback: message received %r" % body.decode())

    # Simulate the situation of reading/processing a complex task that take multiple secs
    time.sleep(body.count(b"."))
    print("[x] callback: message processing is done")

    ch.basic_ack(delivery_tag=method.delivery_tag)
    return


def main(queue_name: str = "hello"):

    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
    channel = connection.channel()

    # We could avoid the below if we were sure the queue already exists, e.g. the producer program was run before.
    # Howeverm we're not yet sure which program to run first.
    channel.queue_declare(queue=queue_name)

    channel.basic_consume(queue=queue_name, auto_ack=False, on_message_callback=callback)

    print("[*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == "__main__":

    # Keep the receiver up until a keybound interruption
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
