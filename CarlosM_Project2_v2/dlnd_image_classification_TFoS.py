# GET DATA
#============================#
import os
import re
import sys
import tarfile

from six.moves import urllib

from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import numpy as np
import tensorflow as tf
import pickle

import random
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from com.yahoo.ml.tf import TFCluster, TFNode
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Set Global Variable for Data Folder Path
cifar10_dataset_folder_path = 'cifar-10-batches-py'

worker_num = ""
job_name = ""
task_index = ""
num_workers = ""

# HELPER
def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    """
    Display Stats of the the dataset
    """
    batch_ids = list(range(1, 6))

    if batch_id not in batch_ids:
        print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
        return None

    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch {}:'.format(batch_id))
    print('Samples: {}'.format(len(features)))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('First 20 Labels: {}'.format(labels[:20]))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    label_names = _load_label_names()

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    plt.axis('off')
    plt.imshow(sample_image)


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the training data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all training data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_training.p')


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def display_image_predictions(features, labels, predictions):
    n_classes = 10
    label_names = _load_label_names()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axies = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        correct_name = label_names[label_id]

        axies[image_i][0].imshow(feature)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])

# Class to Track Download Progress
#=================================#
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

# Check If CIFAR data is available; if Not, then download
#========================================================#
def maybe_download_and_extract():
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    """Download and extract the tarball from Alex's website."""
    dest_directory = "/tmp/cifar10_data"
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)





























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

# Convolutional Neural Network Model
#======================================#
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

# Session Run
#======================================#
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch, x, y, keep_prob ):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict = {x: feature_batch, y: label_batch, keep_prob: keep_probability})

# Print Statistics: Loss and Accuracy
#======================================#
def print_stats(session, feature_batch, label_batch, cost, accuracy,  x, y, keep_prob, valid_features, valid_labels):
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

# Preprocess all data and Save and Train
#=======================================#
def preprocess_data_and_save(epochs, batch_size, keep_probability, ctx):
    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            # Preprocess Training, Validation, and Testing Data
            helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

            # Load the Preprocessed Validation data
            valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

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
            global_step = tf.Variable(0)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost, global_step=global_step)

            # Accuracy
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

            save_model_path = './image_classification'

            # Save Model
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        logdir = TFNode.hdfs_path(ctx, args.model)
        print("tensorflow model path: {0}".format(logdir))
        summary_writer = tf.summary.FileWriter("tensorboard_%d" % (worker_num), graph=tf.get_default_graph())
        sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                 logdir=logdir,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 global_step=global_step,
                                 summary_writer=summary_writer,
                                 saver=saver,
                                 save_model_secs=10)

        print('Training...')
        with sv.managed_session(server.target) as sess:
            # Initializing the variables
            # init_op = tf.global_variables_initializer()
            # sess.run(tf.global_variables_initializer())

            # Training cycle
            for epoch in range(epochs):
                # Loop over all batches
                n_batches = 5
                for batch_i in range(1, n_batches + 1):
                    for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                        train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels, x, y, keep_prob )
#                    print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                    print_stats(sess, batch_features, batch_labels, cost, accuracy, x, y, keep_prob, valid_features, valid_labels )

            # Save Model
            save_path = saver.save(sess, save_model_path)

        # Ask for all the services to stop.
        print("{0} stopping supervisor".format(datetime.now().isoformat()))
        sv.stop()


def test_model(batch_size):
    """
    Test the saved model against the test dataset
    """

    # Set batch size if not already set
    try:
        if batch_size:
            pass
    except NameError:
        batch_size = 64

    save_model_path = './image_classification'
    n_samples = 4
    top_n_predictions = 3

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


# def main():
def main_fun(argv, ctx):
    import tensorflow as tf
    import time

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    cluster_spec = ctx.cluster_spec
    num_workers = len(cluster_spec['worker'])

    # Verify if CIFAR data is available, if not, then download
    maybe_download_and_extract()

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # Hyperparameter Tune Parameters
    epochs = 100
    batch_size = 128
    keep_probability = 0.5

    # Train Model
    preprocess_data_and_save(epochs, batch_size, keep_probability,ctx)

    # Test Model
    test_model(batch_size)


if __name__ == '__main__':
# main_fun("none","none")
    import argparse

    tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs per node.')
    tf.app.flags.DEFINE_boolean('rdma', False, 'Use RDMA between GPUs')

    sc = SparkContext(conf=SparkConf().setAppName("DL_IMAGE_CLASS"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
    num_ps = 1
    tensorboard = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=0)
    parser.add_argument("-f", "--format", help="example format: (csv|pickle|tfr)", choices=["csv", "pickle", "tfr"],
                        default="tfr")
    parser.add_argument("-i", "--images", help="HDFS path to MNIST images in parallelized format")
    parser.add_argument("-l", "--labels", help="HDFS path to MNIST labels in parallelized format")
    parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/test", default="mnist_model")
    parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster (for Spark Standalone)", type=int,
                        default=num_executors)
    parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
    parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=1)
    parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1000)
    parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
    parser.add_argument("-X", "--mode", help="train|inference", default="train")
    parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
    args = parser.parse_args()
    print("args:", args)

    print("{0} ===== Start".format(datetime.now().isoformat()))

    cluster = TFCluster.reserve(sc, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW)
    cluster.start(main_fun, args)
    cluster.shutdown()

    print("{0} ===== Stop".format(datetime.now().isoformat()))
