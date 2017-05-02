# MNIST classifier (convolutional Nenural network) using tensorflow
#
#
# Usage:
# python3.5 -m pip install tensorflow
# python3 mnist_cnn.py
# python3.5 mnist_cnn.py
#
# tensorboard --logdir=summary
#
# https://www.tensorflow.org/get_started/mnist/pros
# JSCH 2017-05-01

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def main():

    FLAGS = flags()

    # Import MNIST data (downloade, extract and interpret the data)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    sess = tf.InteractiveSession()

    # Set parameters (hyperparameters)
    learning_rate = 0.01
    training_iteration = 50
    batch_size = 100
    display_step = 100
    iterrations = 100000

    ## 
    # Build the model
    #

    # Rearrange data into a 4 dimensional tensor (image no./batch, width/height, 
    # height/width, layer deepth). We get an array of images, these images are 
    # represented as 784 (float) arrays, so we reshape them to 28x28 pixel arrays.
    # '-1' will be interpreted as the full size/shape of the input size (no. of 
    # images).

    with tf.name_scope("Reshaping_data") as scope:
        x = tf.placeholder(tf.float32, shape=[None, 784]) # mnist data image of shape 28*28=784
        y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 0-9 digits recognition => 10 classes
        x_image = tf.reshape(x, [-1,28,28,1])
        image_summ = tf.summary.image("Example_images", x_image)


    # Build first convoluFLAGStional layer (width/height, height/width, input 
    # features/layers, output features/layers)
    h_pool1 = cnn_layer(x_image, [5, 5], 8, "layer1_input")

    # Second convolutional layer
    h_pool2 = cnn_layer(h_pool1, [5, 5], 16, "layer2_hidden")

    # Densely (not fully connected, cause of drop out) connected layer
    # NOTE: TensorFlow's tf.nn.dropout op automatically handles scaling neuron 
    # outputs in addition to masking them, so dropout just works without any 
    # additional scaling.

    with tf.name_scope('flattening'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
    h_fc1 = nn_layer(h_pool2_flat, 7*7*16, 1024, 'layer3')

    #with tf.name_scope(layer_name):
        #with tf.name_scope('weights'):
            #W_fc1 = weight_variable([7 * 7 * 16, 1024]) # original 1024 output features
        #with tf.name_scope('bias'):
            #b_fc1 = bias_variable([1024])
        #with tf.name_scope('bias'):
            #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output/Readout layer
    #W_fc2 = weight_variable([1024, 10])
    #b_fc2 = bias_variable([10])

    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y_conv = nn_layer(h_fc1_drop, 1024, 10, 'layer3')

    with tf.name_scope('cost'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('sgd_adam'):
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_step = optimizer.minimize(cross_entropy)
        #gradient = tf.gradients(cross_entropy,)
        #tf.summary.scalar('pred_error', 1-accuracy)
        
    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('pred_error', 1-accuracy)

    #arr_s = np.rint(iterrations/display_step)
    arr_s = int(iterrations/display_step)
    train_accuracy = np.zeros(arr_s)
    valid_accuracy = np.zeros(arr_s)
    cost = np.zeros(arr_s)


    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()



    sess.run(tf.global_variables_initializer())
    for i in range(iterrations):
        batch = mnist.train.next_batch(50)

        summary,_ = sess.run([merged, train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        train_writer.add_summary(summary, i)

        if i%display_step == 0:
            arr_no = int(i/display_step)
            #cost[arr_no] = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            #train_accuracy[arr_no] = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            cost[arr_no], train_accuracy[arr_no] = sess.run([cross_entropy, accuracy],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary, valid_accuracy[arr_no] = sess.run([merged, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
            
            valid_writer.add_summary(summary, i)
            
            train_writer.flush() # Don't forget this command! It makes sure Python writes the summaries to the log-file
            valid_writer.flush()
            
            print("step {:05d}, cost {:4f}, training accuracy {:4f}, test accuracy {:4f}".format(i, cost[arr_no], train_accuracy[arr_no], valid_accuracy[arr_no]))

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


    x_image_ = x_image.eval(feed_dict={x: mnist.test.images[1:2]})
    h_pool1_ = h_pool1.eval(feed_dict={x_image: x_image_})
    h_pool2_ = h_pool2.eval(feed_dict={h_pool1: h_pool1_})

    x_image_ = x_image_[0]
    h_pool1_ = h_pool1_[0]
    h_pool2_ = h_pool2_[0]

    #x_image_ = np.squeeze(x_image_)
    #h_pool1_ = np.squeeze(h_pool1_)
    #h_pool2_ = np.squeeze(h_pool2_)

    plt.gray()
    plt.subplot(4,1,1)
    plt.imshow(np.squeeze(x_image_))

    for i in range(h_pool1_.shape[2]):
        plt.subplot(4,8,8+1+i)
        plt.imshow(np.squeeze(h_pool1_[:,:,i]))

    for i in range(h_pool2_.shape[2]):
        plt.subplot(4,8,16+1+i)
        img = np.squeeze(h_pool2_[:,:,i])
        plt.imshow(img)

    plt.gray()
    plt.show()

    itterations = np.linspace(1, iterrations/display_step, iterrations/display_step)
    plt.plot(cost,'r-',1-train_accuracy,'g-',1-valid_accuracy,'b-')
    plt.show()


class flags():
    def __init__(self, summaries_dir='summary/',data_dir='data/'):
        self.summaries_dir = summaries_dir
        self.data_dir = data_dir

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def cnn_layer(input_tensor, conv_dim, output_depth, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple convolutional neural net layer.

    It does a convolution, applies an activation function (default: rectifier 
    linear), and adds bias per output layer (depth). It also sets up name
    scoping so that the resultant graph is easy to read, and adds a number of
    summary ops.
    """

    #tf.shape(input_tensor)[0]

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            conv_dim.append(input_tensor.get_shape().as_list()[3])
            conv_dim.append(output_depth)
            W_conv = weight_variable(conv_dim)
            variable_summaries(W_conv)
        with tf.name_scope('bias'):
            b_conv = bias_variable([conv_dim[3]])
            variable_summaries(W_conv)
        with tf.name_scope('conv_relu_bias'):
            h_relu = tf.nn.relu(conv2d(input_tensor, W_conv) + b_conv)
        with tf.name_scope('max_pool'):
            h_pool = max_pool_2x2(h_relu)
    return h_pool

if __name__ == "__main__":
    main()
