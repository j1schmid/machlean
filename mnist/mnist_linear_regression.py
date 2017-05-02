# A simple MNIST classifier (linear regression) with tensorflow
#
# Based on https://github.com/llSourcell/tensorflow_demo.git board.py
# some modification for python 3 (some deprecated/renamed functions and print ...)
#
# See also:
# ---------
#
# Siraj Raval's video/tutorial
#   https://www.youtube.com/watch?v=2FmcHiLCwTU
#
# TensorFlow official Tutorial
#   http://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/
#
# manually handle gradients
#   https://www.tensorflow.org/api_docs/python/tf/train/Optimizer
#   https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
#       
#
#
# Usage:
# python3.5 -m pip install tensorflow
# python3 board.py
# python3.5 board.py
#
# tensorboard --logdir=summary
# 
# JSCH 2017-04-23

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def main():

    print("TensorFlow version ", tf.__version__)

    mnist = input_data.read_data_sets("data/", one_hot=True)

    print("mnist data set read:")
    print('%04d' % mnist.train.num_examples, "training examples")
    print('%04d' % mnist.validation.num_examples, "validation examples")
    print('%04d' % mnist.test.num_examples, "test examples")


    # Set parameters
    learning_rate = 0.01
    training_iteration = 30
    batch_size = 100
    display_step = 2

    # TF graph input
    x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

    # Create a model

    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    with tf.name_scope("Wx_b") as scope:
        # Construct a linear model
        model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        
    # Add summary ops to collect data
    w_h = tf.summary.histogram("weights", W)
    b_h = tf.summary.histogram("biases", b)

    # More name scopes will clean up graph representation
    with tf.name_scope("cost_function") as scope:
        # Minimize error using cross entropy
        # Cross entropy
        cost_function = -tf.reduce_sum(y*tf.log(model))
        
        # Create a summary to monitor the cost function, use the mean of the cross
        # entropy to compare it with other batch sizes
        tf.summary.scalar("cost_function", -tf.reduce_mean(y*tf.log(model)))

    with tf.name_scope("train") as scope:
        # Gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        
        # Compute the gradients for a list of variables.
        grads_and_vars = optimizer.compute_gradients(cost_function)
        
        # grads_and_vars is a list of tuples (gradient, variable). Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        #capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
        #capped_grads_and_vars = grads_and_vars
        grads_and_vars = optimizer.compute_gradients(cost_function)
        capped_grads_and_vars = grads_and_vars
        
        # Ask the optimizer to apply the capped gradients.
        train_step = optimizer.apply_gradients(capped_grads_and_vars)
        
    with tf.name_scope("gradient") as scope:
        grad_norm = tf.global_norm(grads_and_vars)
        grad_norm_s = tf.summary.scalar("norm", grad_norm)
       
        
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Merge all summaries into a single operator
    merged_summary_op = tf.summary.merge_all()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        print("iter ; train cost ; valid cost")
        
        # Change this to a location on your computer
        test_writer = tf.summary.FileWriter('summary/test')
        train_writer = tf.summary.FileWriter('summary/train', graph=sess.graph)

        avg_vcost = 0. # average validation cost

        # Training cycle
        for iteration in range(training_iteration):
            avg_cost = 0. # average training cost
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Fit training using batch data, 
                
                # Compute the average loss
                
                # write summary, compute the loss/cost, fit training using batch data
                #sess.run(, feed_dict={x: batch_xs, y: batch_ys})
                summary_str, cost, _ = sess.run([merged_summary_op, cost_function, train_step], feed_dict={x: batch_xs, y: batch_ys})
                
                # compute the average cost
                avg_cost += cost/(batch_size*total_batch)
                
                # Write logs for each iteration
                #summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
                train_writer.add_summary(summary_str, iteration*total_batch + i)
                
            summary_str = sess.run(merged_summary_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
            test_writer.add_summary(summary_str, iteration*total_batch)

            # Display logs per iteration step
            avg_vcost += sess.run(cost_function, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})/(mnist.validation.num_examples*display_step)
            if iteration % display_step == 0:
                print("{:05d} ;   {:.4f}   ;   {:.4f}".format((iteration + 1),avg_cost,avg_vcost))
                avg_vcost = 0.
                
                
        print("Tuning completed!")

        # Test the model
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
        print("Accuracy test set:      ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print("   \"     validation set:", accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels}))
        print("   \"     training set:  ", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))

        print("Cost     test set:      ", cost_function.eval({x: mnist.test.images, y: mnist.test.labels})/mnist.test.num_examples  )
        print(" \"       validation set:", cost_function.eval({x: mnist.validation.images, y: mnist.validation.labels})/mnist.validation.num_examples)
        print(" \"       training set:  ", cost_function.eval({x: mnist.train.images, y: mnist.train.labels})/mnist.train.num_examples  )


if __name__ == "__main__":
    main()
