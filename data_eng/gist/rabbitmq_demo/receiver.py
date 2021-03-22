"""
https://www.rabbitmq.com/tutorials/tutorial-one-python.html

brew services start rabbitmq

"""
import pika


def callback(ch, method, properties, body):
    """Standard callback to receive messages from a queue"""
    print("[x] Received %r" % body)
    return


def main(queue_name: str = "hello"):

    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
    channel = connection.channel()

    # We could avoid the below if we were sure  the queue already exists. For example if sender.py program was run before.
    # But we're not yet sure which program to run first.
    channel.queue_declare(queue=queue_name)

    channel.basic_consume(queue="hello", auto_ack=True, on_message_callback=callback)

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
