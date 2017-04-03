# GET DATA
#============================#
import pickle
import random
import tarfile
from os.path import isfile, isdir
from urllib.request import urlretrieve
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from CarlosM_Project2_v1 import helper

# Set Global Variable for Data Folder Path
cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Download Data
#============================#
def download_images():

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile('cifar-10-python.tar.gz'):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'cifar-10-python.tar.gz',
                pbar.hook)

    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open('cifar-10-python.tar.gz') as tar:
            tar.extractall()
            tar.close()

# Normalize Data
#============================#
def normalize(data):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # Convert all data to Float32 in preparation for normalization
    data = data.astype('float32')

    # divide by Max Value (255) to normalize
    data /= data.max()
    return data

# One Hot Encode
#============================#
def one_hot_encode(labelsArray, total_lbs=10):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # Array of zeros with the given shape (rows, columns)
    result = np.zeros((len(labelsArray), total_lbs))

    # set the number 1 to the position specified by labelsArray
    result[np.arange(len(labelsArray)), labelsArray] = 1
    return result

# Image Input
#============================#
def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0],image_shape[1],image_shape[2]], name = 'x')

# Label Input
#============================#
def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], name = 'y')

# Dropout keep probability Input
#============================#
def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name = "keep_prob")

# Convolution and Max Pooling Layer
#==================================#
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # Weight and bias
    # Get the Shape of the Tensor
    x = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], x[3], conv_num_outputs], stddev=5e-2, mean=0.01))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    conv1 = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, bias)

    # Batch Normalization Tech. to normalize output of every conv layer
    conv1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    return conv1

# Flatten Tensor from 4D to 2D
#==================================#
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # Get the Shape of the Tensor
    x=x_tensor.get_shape().as_list()

    # Multiply dimensions and reshape infering
    fc1 = tf.reshape(x_tensor, [-1,x[1]*x[2]*x[3]])
    return fc1

# Apply Fully Connected Layer to Tensor
#======================================#
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # Get the Shape of the Tensor
    x = x_tensor.get_shape().as_list()

    # Create Weights and Bias using tensor shape and outputs
    weight = tf.Variable(tf.truncated_normal([x[1], num_outputs], stddev=5e-2, mean=0.01))
    bias = tf.Variable(tf.zeros(num_outputs))

    # Multiply and Add Bias and Rectify
    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))

# Output Layer
#======================================#
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # Get the Shape of the Tensor
    x = x_tensor.get_shape().as_list()

    # Create Weights and Bias using tensor shape and outputs
    weight = tf.Variable(tf.truncated_normal([x[1], num_outputs], stddev=5e-2, mean=0.01))
    bias = tf.Variable(tf.zeros(num_outputs))

    # Output Layer - class prediction
    out = tf.add(tf.matmul(x_tensor, weight), bias)
    return out

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    # conv2d_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv1 = conv2d_maxpool(x, 32, (5, 5), (1, 1), (3, 3), (2, 2))
    conv2 = conv2d_maxpool(conv1, 64, (5, 5), (1, 1), (3, 3), (2, 2))
    conv3 = conv2d_maxpool(conv2, 128, (5, 5), (1, 1), (3, 3), (2, 2))
    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    conv2_f = flatten(conv3)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc1 = fully_conn(conv2_f, 512)
    fc2 = fully_conn(fc1, 1024)
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    ol1 = output(fc2, 10)

    # Added Dropout after output layer; tried it before other layer
    # but the network died.
    do = tf.nn.dropout(ol1, keep_prob)

    # TODO: return output
    return do

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    feed_dict = {
        x:feature_batch,
        y:label_batch,
        keep_prob:keep_probability
    }
    session.run(optimizer, feed_dict)

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # Calculate batch loss and accuracy
    loss = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    valid_acc = session.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.})

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss,valid_acc))

def test_model(batch_size, save_model_path, n_samples, top_n_predictions):
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels,
                                                                                   batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(
            zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)

def main():

    download_images()

    # Preprocess all data and save
    # ==============================#

    # Preprocess Training, Validation, and Testing Data
    helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

    # Load the Preprocessed Validation data
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

    # TODO: Tune Parameters
    epochs = 100
    batch_size = 128
    keep_probability = 0.5

    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = neural_net_image_input((32, 32, 3))
    y = neural_net_label_input(10)
    keep_prob = neural_net_keep_prob_input()

    # Model
    logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    save_model_path = './image_classification'

    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)

    # Set batch size if not already set
    try:
        if batch_size:
            pass
    except NameError:
        batch_size = 64

    save_model_path = './image_classification'
    n_samples = 4
    top_n_predictions = 3

    test_model(batch_size, save_model_path, n_samples, top_n_predictions)

if __name__ == '__main__':
    main()