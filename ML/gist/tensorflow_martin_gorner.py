
"""
https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd

# Usage
  $ python3 ./tensorflow_martin_gorner.py
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from dsutils.generic import createLogger

logger = createLogger(__name__) 

def main():
    
    # Parameters
    num_images = None
    num_columns = 28
    num_rows = 28
    num_classes = 10

    # Initialize
    X = tf.placeholder(tf.float32, [num_images, num_rows, num_columns, 1])
    W = tf.Variable(tf.zeros([num_rows * num_columns, num_classes]))
    b = tf.Variable(tf.zeros(num_classes))
    init = tf.global_variables_initializer()
    logger.info("Successfully initialized")

    # Set up the model
    Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, num_rows * num_columns]), W) + b)
    Y_ = tf.placeholder(tf.float32, [None, num_classes])  # placeholder for true labels
    
    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))  # Loss function
    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)
    
    # Performance metrics
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    logger.info("Successfully set up the model")
    
    # Data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    ##########
    # Start the training
    ##########
    sess = tf.Session()
    sess.run(init)

    for idx in range(1000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, 
                      Y_: batch_Y}
        
        # Train
        sess.run(train_step, feed_dict=train_data)
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        logger.info("Train set: accuracy = %.2f, cross entropu = %.2f" %(a, c))
        
        # Test
        test_data = {X: mnist.test.images, 
                     Y_: mnist.test.lables}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        logger.info("Test set: accuracy = %.2f, cross entropu = %.2f" %(a, c))


if __name__ == "__main__":
    main()
    
    # 
    logger.info("ALL DONE!")
