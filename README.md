

# Image Classification using Convolutional Neural Networks
## Udacity Project #2
This is my submission to Project #2 of Udacity's Deep Learning Fundation course.
On this project I created a Convolutional Neural Network (CNN) to classify images of the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) image set.

Libraries & Tools Used:
* Tensor Flow
* Jupyter Notebook
* Numpy
* Conda

Training VM:
* AWS p2.xlarge instances (GPU)

## Download the Data
On this project we classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of airplanes, dogs, cats, and other objects. Before training a model we need to preprocess the images and then train a convolutional neural network on all the samples. The images are normalized and the labels are one-hot encoded.  For the model I added convolutional, max pooling, dropout, and fully connected layers.
[CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).


```python
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

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


tests.test_folder_path(cifar10_dataset_folder_path)
```

    CIFAR-10 Dataset: 171MB [01:12, 2.37MB/s]                              


    All files found!


## CIFAR-10 Data Set
The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

# Dataset Example
batch_id = 1
sample_id = 6
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```

    
    Stats of batch 1:
    Samples: 10000
    Label Counts: {0: 1005, 1: 974, 2: 1032, 3: 1016, 4: 999, 5: 937, 6: 1030, 7: 1001, 8: 1025, 9: 981}
    First 20 Labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7, 2, 9, 9, 9, 3, 2, 6]
    
    Example of Image 6:
    Image - Min Value: 7 Max Value: 249
    Image - Shape: (32, 32, 3)
    Label - Label Id: 2 Name: bird



![png](output_3_1.png)


## Preprocess Functions
### Normalize
This is the `normalize` function to take in image data, `x`, and return it as a normalized Numpy array. The values are in the range of 0 to 1, inclusive.  The return object is the same shape as `x`.


```python
def normalize(data):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # Convert all data to Float32 in preparation for normalization
    data = data.astype('float32')
    # Check is data is Not Normalized before attempting normalization
    if data.max() > 1.0:
        # divide by 255 to normalize
        data/=255.0
    return data


"""
Function Test
"""
tests.test_normalize(normalize)
```

    Tests Passed


### One-hot encode
Implementation of the `one_hot_encode` function. The input, `x`, are a list of labels.  The function return the list of labels as One-Hot encoded Numpy array.  The possible values for labels are 0 to 9.


```python
def one_hot_encode(labelsArray, total_lbs=10):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # Array of zeros with the given shape (rows, columns)
    # Rows (x) = numpyArray.shape[0]
    # Columns (y) = total_lbs (Total Labels)
    result = np.zeros((len(labelsArray), total_lbs))
    
    # set the number 1 to the position specified by numpyArray
    # for every row (x) and column (value of numpyArray)
    # Effectively creating a matrix with #1 on the position of the label
    result[np.arange(len(labelsArray)), labelsArray] = 1
    return result

"""
Function Test
"""
tests.test_one_hot_encode(one_hot_encode)
```

    Tests Passed


## Preprocess all the data and save it
Preprocess all the CIFAR-10 data and save it to file. 10% of the training data is used for validation.


```python
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
```


```python
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
```

# Build the network
For the neural network, each layer is built into a function.

### Input
Read the image data, one-hot encoded labels, and dropout keep probability. 


```python
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0],image_shape[1],image_shape[2]], name = 'x')

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], name = 'y')

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name = "keep_prob")

"""
Function Tests
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
```

    Image Input Tests Passed.
    Label Input Tests Passed.
    Keep Prob Tests Passed.


### Convolution and Max Pooling Layer


```python
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

"""
Function Test
"""
tests.test_con_pool(conv2d_maxpool)
```

    Tests Passed


### Flatten Layer
Function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output is of shape (*Batch Size*, *Flattened Image Size*).


```python
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

"""
Function Test
"""
tests.test_flatten(flatten)
```

    Tests Passed


### Fully-Connected Layer
`fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*).


```python
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

    # Multiply and Add Bias and Activation
    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))

"""
Function Test
"""
tests.test_fully_conn(fully_conn)
```

    Tests Passed


### Output Layer
`output` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*).


```python
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
    weight = tf.Variable(tf.truncated_normal([x[1], num_outputs]))
    bias = tf.Variable(tf.zeros(num_outputs))
    
    # Output Layer - class prediction 
    out = tf.add(tf.matmul(x_tensor, weight), bias)
    return out

"""
Function Test
"""
tests.test_output(output)
```

    Tests Passed


### Create Convolutional Model
`conv_net` function to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # conv2d_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv1 = conv2d_maxpool(x, 32, (5,5), (1,1), (3,3), (2,2))
    conv2 = conv2d_maxpool(conv1, 64, (5,5), (1,1), (3,3), (2,2))
    conv2 = conv2d_maxpool(conv1, 128, (5,5), (1,1), (3,3), (2,2))
    
    #   flatten(x_tensor)
    conv2_f = flatten(conv2)

    #   fully_conn(x_tensor, num_outputs)
    fc1 = fully_conn(conv2_f, 512)
    fc2 = fully_conn(fc1, 1024)

    #   output(x_tensor, num_outputs)
    ol1 = output(fc2, 10)

    # Added Dropout after output layer; tried it before other layer
    # but the network died.
    do = tf.nn.dropout(ol1, keep_prob)
    
    return do

"""
Function Test
"""

##############################
## Build the Neural Network ##
##############################

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

tests.test_conv_net(conv_net)
```

    Neural Network Built!


## Train the Neural Network
### Single Optimization
`train_neural_network` function to do a single optimization.


```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
    
"""
Function Test
"""
tests.test_train_nn(train_neural_network)
```

    Tests Passed


### Show Stats
`print_stats` function to print loss and validation accuracy.


```python
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
```

### Hyperparameters
Parameters to Tune:


```python
# Tune Parameters
epochs = 100
batch_size = 128
keep_probability = 0.5
```

### Fully Train the Model
Train the model using all five batches.


```python
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
```

    Training...
    Epoch  1, CIFAR-10 Batch 1:  Loss:   144.5920 Validation Accuracy: 0.179600
    Epoch  1, CIFAR-10 Batch 2:  Loss:   244.7793 Validation Accuracy: 0.141200
    Epoch  1, CIFAR-10 Batch 3:  Loss:   280.2766 Validation Accuracy: 0.126200
    Epoch  1, CIFAR-10 Batch 4:  Loss:    88.6685 Validation Accuracy: 0.195400
    Epoch  1, CIFAR-10 Batch 5:  Loss:   192.2382 Validation Accuracy: 0.206000
    Epoch  2, CIFAR-10 Batch 1:  Loss:   215.4647 Validation Accuracy: 0.201600
    Epoch  2, CIFAR-10 Batch 2:  Loss:   146.8320 Validation Accuracy: 0.197200
    Epoch  2, CIFAR-10 Batch 3:  Loss:   115.4242 Validation Accuracy: 0.255000
    Epoch  2, CIFAR-10 Batch 4:  Loss:   120.8510 Validation Accuracy: 0.198000
    Epoch  2, CIFAR-10 Batch 5:  Loss:   101.4128 Validation Accuracy: 0.190600
    Epoch  3, CIFAR-10 Batch 1:  Loss:   143.6717 Validation Accuracy: 0.263200
    Epoch  3, CIFAR-10 Batch 2:  Loss:   217.9221 Validation Accuracy: 0.194000
    Epoch  3, CIFAR-10 Batch 3:  Loss:   160.2292 Validation Accuracy: 0.165600
    Epoch  3, CIFAR-10 Batch 4:  Loss:   173.7155 Validation Accuracy: 0.220400
    Epoch  3, CIFAR-10 Batch 5:  Loss:   178.7390 Validation Accuracy: 0.262000
    Epoch  4, CIFAR-10 Batch 1:  Loss:   114.0415 Validation Accuracy: 0.176000
    Epoch  4, CIFAR-10 Batch 2:  Loss:   131.7612 Validation Accuracy: 0.199600
    Epoch  4, CIFAR-10 Batch 3:  Loss:    72.4614 Validation Accuracy: 0.209000
    Epoch  4, CIFAR-10 Batch 4:  Loss:    81.5187 Validation Accuracy: 0.233600
    Epoch  4, CIFAR-10 Batch 5:  Loss:    99.3658 Validation Accuracy: 0.231000
    Epoch  5, CIFAR-10 Batch 1:  Loss:    58.4843 Validation Accuracy: 0.254000
    Epoch  5, CIFAR-10 Batch 2:  Loss:    97.1832 Validation Accuracy: 0.233200
    Epoch  5, CIFAR-10 Batch 3:  Loss:   138.2004 Validation Accuracy: 0.243400
    Epoch  5, CIFAR-10 Batch 4:  Loss:   282.2009 Validation Accuracy: 0.271600
    Epoch  5, CIFAR-10 Batch 5:  Loss:   113.4337 Validation Accuracy: 0.237000
    Epoch  6, CIFAR-10 Batch 1:  Loss:    89.8026 Validation Accuracy: 0.304400
    Epoch  6, CIFAR-10 Batch 2:  Loss:    79.5671 Validation Accuracy: 0.281600
    Epoch  6, CIFAR-10 Batch 3:  Loss:   121.6496 Validation Accuracy: 0.263200
    Epoch  6, CIFAR-10 Batch 4:  Loss:    41.2070 Validation Accuracy: 0.300800
    Epoch  6, CIFAR-10 Batch 5:  Loss:   240.9617 Validation Accuracy: 0.237400
    Epoch  7, CIFAR-10 Batch 1:  Loss:    40.7588 Validation Accuracy: 0.250000
    Epoch  7, CIFAR-10 Batch 2:  Loss:    72.9107 Validation Accuracy: 0.310400
    Epoch  7, CIFAR-10 Batch 3:  Loss:   118.7067 Validation Accuracy: 0.239800
    Epoch  7, CIFAR-10 Batch 4:  Loss:    54.2522 Validation Accuracy: 0.253800
    Epoch  7, CIFAR-10 Batch 5:  Loss:   114.1559 Validation Accuracy: 0.272400
    Epoch  8, CIFAR-10 Batch 1:  Loss:    68.7883 Validation Accuracy: 0.215800
    Epoch  8, CIFAR-10 Batch 2:  Loss:   124.7608 Validation Accuracy: 0.230000
    Epoch  8, CIFAR-10 Batch 3:  Loss:    85.4661 Validation Accuracy: 0.301200
    Epoch  8, CIFAR-10 Batch 4:  Loss:    21.9460 Validation Accuracy: 0.319400
    Epoch  8, CIFAR-10 Batch 5:  Loss:   109.1754 Validation Accuracy: 0.303200
    Epoch  9, CIFAR-10 Batch 1:  Loss:    53.5675 Validation Accuracy: 0.326200
    Epoch  9, CIFAR-10 Batch 2:  Loss:    70.8362 Validation Accuracy: 0.283200
    Epoch  9, CIFAR-10 Batch 3:  Loss:    43.2077 Validation Accuracy: 0.331000
    Epoch  9, CIFAR-10 Batch 4:  Loss:    59.2074 Validation Accuracy: 0.233800
    Epoch  9, CIFAR-10 Batch 5:  Loss:    97.5286 Validation Accuracy: 0.308200
    Epoch 10, CIFAR-10 Batch 1:  Loss:   106.8098 Validation Accuracy: 0.259600
    Epoch 10, CIFAR-10 Batch 2:  Loss:   112.1532 Validation Accuracy: 0.244400
    Epoch 10, CIFAR-10 Batch 3:  Loss:    63.1969 Validation Accuracy: 0.277400
    Epoch 10, CIFAR-10 Batch 4:  Loss:    30.3070 Validation Accuracy: 0.316800
    Epoch 10, CIFAR-10 Batch 5:  Loss:    56.8260 Validation Accuracy: 0.347200
    Epoch 11, CIFAR-10 Batch 1:  Loss:    26.9426 Validation Accuracy: 0.322800
    Epoch 11, CIFAR-10 Batch 2:  Loss:    82.8732 Validation Accuracy: 0.250400
    Epoch 11, CIFAR-10 Batch 3:  Loss:    62.9233 Validation Accuracy: 0.283200
    Epoch 11, CIFAR-10 Batch 4:  Loss:    60.0825 Validation Accuracy: 0.354200
    Epoch 11, CIFAR-10 Batch 5:  Loss:   101.7196 Validation Accuracy: 0.291800
    Epoch 12, CIFAR-10 Batch 1:  Loss:    54.0462 Validation Accuracy: 0.309600
    Epoch 12, CIFAR-10 Batch 2:  Loss:    91.6873 Validation Accuracy: 0.323200
    Epoch 12, CIFAR-10 Batch 3:  Loss:    63.1348 Validation Accuracy: 0.326800
    Epoch 12, CIFAR-10 Batch 4:  Loss:    53.0382 Validation Accuracy: 0.327800
    Epoch 12, CIFAR-10 Batch 5:  Loss:   100.6232 Validation Accuracy: 0.304000
    Epoch 13, CIFAR-10 Batch 1:  Loss:    33.0407 Validation Accuracy: 0.353400
    Epoch 13, CIFAR-10 Batch 2:  Loss:   207.8501 Validation Accuracy: 0.315600
    Epoch 13, CIFAR-10 Batch 3:  Loss:    80.3235 Validation Accuracy: 0.335200
    Epoch 13, CIFAR-10 Batch 4:  Loss:    24.0340 Validation Accuracy: 0.326600
    Epoch 13, CIFAR-10 Batch 5:  Loss:   132.9572 Validation Accuracy: 0.277600
    Epoch 14, CIFAR-10 Batch 1:  Loss:    27.2212 Validation Accuracy: 0.322000
    Epoch 14, CIFAR-10 Batch 2:  Loss:    74.3011 Validation Accuracy: 0.313200
    Epoch 14, CIFAR-10 Batch 3:  Loss:    64.2017 Validation Accuracy: 0.328800
    Epoch 14, CIFAR-10 Batch 4:  Loss:    52.9062 Validation Accuracy: 0.331000
    Epoch 14, CIFAR-10 Batch 5:  Loss:    75.0369 Validation Accuracy: 0.343000
    Epoch 15, CIFAR-10 Batch 1:  Loss:    23.7709 Validation Accuracy: 0.379600
    Epoch 15, CIFAR-10 Batch 2:  Loss:    44.0940 Validation Accuracy: 0.353400
    Epoch 15, CIFAR-10 Batch 3:  Loss:    78.9234 Validation Accuracy: 0.348600
    Epoch 15, CIFAR-10 Batch 4:  Loss:    66.0491 Validation Accuracy: 0.312600
    Epoch 15, CIFAR-10 Batch 5:  Loss:    34.8804 Validation Accuracy: 0.377800
    Epoch 16, CIFAR-10 Batch 1:  Loss:    30.6415 Validation Accuracy: 0.307000
    Epoch 16, CIFAR-10 Batch 2:  Loss:    56.0044 Validation Accuracy: 0.325800
    Epoch 16, CIFAR-10 Batch 3:  Loss:    12.6109 Validation Accuracy: 0.378800
    Epoch 16, CIFAR-10 Batch 4:  Loss:    48.9802 Validation Accuracy: 0.372000
    Epoch 16, CIFAR-10 Batch 5:  Loss:    42.4325 Validation Accuracy: 0.300800
    Epoch 17, CIFAR-10 Batch 1:  Loss:    26.2347 Validation Accuracy: 0.336400
    Epoch 17, CIFAR-10 Batch 2:  Loss:    94.1761 Validation Accuracy: 0.345000
    Epoch 17, CIFAR-10 Batch 3:  Loss:    36.9028 Validation Accuracy: 0.342000
    Epoch 17, CIFAR-10 Batch 4:  Loss:    46.3676 Validation Accuracy: 0.365600
    Epoch 17, CIFAR-10 Batch 5:  Loss:    19.7045 Validation Accuracy: 0.407000
    Epoch 18, CIFAR-10 Batch 1:  Loss:    40.4953 Validation Accuracy: 0.254800
    Epoch 18, CIFAR-10 Batch 2:  Loss:    34.3107 Validation Accuracy: 0.375200
    Epoch 18, CIFAR-10 Batch 3:  Loss:    20.7018 Validation Accuracy: 0.365000
    Epoch 18, CIFAR-10 Batch 4:  Loss:    48.9497 Validation Accuracy: 0.332000
    Epoch 18, CIFAR-10 Batch 5:  Loss:    30.2418 Validation Accuracy: 0.377800
    Epoch 19, CIFAR-10 Batch 1:  Loss:    88.2027 Validation Accuracy: 0.352600
    Epoch 19, CIFAR-10 Batch 2:  Loss:    35.2203 Validation Accuracy: 0.371200
    Epoch 19, CIFAR-10 Batch 3:  Loss:    33.6955 Validation Accuracy: 0.364200
    Epoch 19, CIFAR-10 Batch 4:  Loss:    24.6600 Validation Accuracy: 0.368200
    Epoch 19, CIFAR-10 Batch 5:  Loss:    34.6328 Validation Accuracy: 0.353400
    Epoch 20, CIFAR-10 Batch 1:  Loss:    33.5882 Validation Accuracy: 0.318800
    Epoch 20, CIFAR-10 Batch 2:  Loss:    37.1797 Validation Accuracy: 0.355200
    Epoch 20, CIFAR-10 Batch 3:  Loss:    18.6388 Validation Accuracy: 0.378000
    Epoch 20, CIFAR-10 Batch 4:  Loss:    15.7321 Validation Accuracy: 0.394800
    Epoch 20, CIFAR-10 Batch 5:  Loss:    18.0132 Validation Accuracy: 0.367400
    Epoch 21, CIFAR-10 Batch 1:  Loss:    10.3918 Validation Accuracy: 0.403600
    Epoch 21, CIFAR-10 Batch 2:  Loss:    54.7764 Validation Accuracy: 0.303400
    Epoch 21, CIFAR-10 Batch 3:  Loss:    26.8143 Validation Accuracy: 0.370000
    Epoch 21, CIFAR-10 Batch 4:  Loss:    18.1459 Validation Accuracy: 0.313400
    Epoch 21, CIFAR-10 Batch 5:  Loss:    27.7423 Validation Accuracy: 0.376400
    Epoch 22, CIFAR-10 Batch 1:  Loss:    25.3703 Validation Accuracy: 0.348400
    Epoch 22, CIFAR-10 Batch 2:  Loss:    18.9215 Validation Accuracy: 0.374600
    Epoch 22, CIFAR-10 Batch 3:  Loss:    29.1799 Validation Accuracy: 0.362600
    Epoch 22, CIFAR-10 Batch 4:  Loss:     6.2573 Validation Accuracy: 0.402600
    Epoch 22, CIFAR-10 Batch 5:  Loss:    19.9332 Validation Accuracy: 0.384400
    Epoch 23, CIFAR-10 Batch 1:  Loss:    34.0209 Validation Accuracy: 0.377400
    Epoch 23, CIFAR-10 Batch 2:  Loss:    25.7399 Validation Accuracy: 0.344400
    Epoch 23, CIFAR-10 Batch 3:  Loss:    27.6721 Validation Accuracy: 0.357000
    Epoch 23, CIFAR-10 Batch 4:  Loss:    32.2387 Validation Accuracy: 0.350400
    Epoch 23, CIFAR-10 Batch 5:  Loss:    31.3015 Validation Accuracy: 0.347200
    Epoch 24, CIFAR-10 Batch 1:  Loss:    26.6810 Validation Accuracy: 0.371000
    Epoch 24, CIFAR-10 Batch 2:  Loss:    21.7396 Validation Accuracy: 0.397400
    Epoch 24, CIFAR-10 Batch 3:  Loss:     4.8055 Validation Accuracy: 0.388800
    Epoch 24, CIFAR-10 Batch 4:  Loss:     5.7480 Validation Accuracy: 0.374600
    Epoch 24, CIFAR-10 Batch 5:  Loss:    27.5901 Validation Accuracy: 0.388600
    Epoch 25, CIFAR-10 Batch 1:  Loss:     8.1964 Validation Accuracy: 0.423000
    Epoch 25, CIFAR-10 Batch 2:  Loss:    19.9796 Validation Accuracy: 0.364600
    Epoch 25, CIFAR-10 Batch 3:  Loss:     3.7691 Validation Accuracy: 0.415800
    Epoch 25, CIFAR-10 Batch 4:  Loss:     2.8239 Validation Accuracy: 0.403800
    Epoch 25, CIFAR-10 Batch 5:  Loss:    17.7497 Validation Accuracy: 0.412200
    Epoch 26, CIFAR-10 Batch 1:  Loss:    16.8212 Validation Accuracy: 0.351200
    Epoch 26, CIFAR-10 Batch 2:  Loss:    12.0919 Validation Accuracy: 0.393400
    Epoch 26, CIFAR-10 Batch 3:  Loss:     6.6443 Validation Accuracy: 0.402000
    Epoch 26, CIFAR-10 Batch 4:  Loss:    13.9184 Validation Accuracy: 0.317000
    Epoch 26, CIFAR-10 Batch 5:  Loss:    21.9917 Validation Accuracy: 0.381200
    Epoch 27, CIFAR-10 Batch 1:  Loss:     6.2449 Validation Accuracy: 0.399000
    Epoch 27, CIFAR-10 Batch 2:  Loss:    22.5307 Validation Accuracy: 0.343400
    Epoch 27, CIFAR-10 Batch 3:  Loss:     8.8485 Validation Accuracy: 0.408800
    Epoch 27, CIFAR-10 Batch 4:  Loss:     3.6252 Validation Accuracy: 0.406400
    Epoch 27, CIFAR-10 Batch 5:  Loss:    20.7076 Validation Accuracy: 0.382400
    Epoch 28, CIFAR-10 Batch 1:  Loss:    10.9915 Validation Accuracy: 0.400000
    Epoch 28, CIFAR-10 Batch 2:  Loss:     8.3838 Validation Accuracy: 0.390000
    Epoch 28, CIFAR-10 Batch 3:  Loss:     8.6642 Validation Accuracy: 0.397800
    Epoch 28, CIFAR-10 Batch 4:  Loss:     3.2386 Validation Accuracy: 0.381000
    Epoch 28, CIFAR-10 Batch 5:  Loss:    31.4968 Validation Accuracy: 0.369000
    Epoch 29, CIFAR-10 Batch 1:  Loss:     9.5839 Validation Accuracy: 0.373000
    Epoch 29, CIFAR-10 Batch 2:  Loss:    10.6545 Validation Accuracy: 0.373600
    Epoch 29, CIFAR-10 Batch 3:  Loss:     3.6832 Validation Accuracy: 0.434200
    Epoch 29, CIFAR-10 Batch 4:  Loss:     2.4861 Validation Accuracy: 0.379000
    Epoch 29, CIFAR-10 Batch 5:  Loss:    16.0673 Validation Accuracy: 0.410000
    Epoch 30, CIFAR-10 Batch 1:  Loss:     8.3772 Validation Accuracy: 0.360000
    Epoch 30, CIFAR-10 Batch 2:  Loss:     7.2559 Validation Accuracy: 0.425200
    Epoch 30, CIFAR-10 Batch 3:  Loss:     5.1744 Validation Accuracy: 0.414000
    Epoch 30, CIFAR-10 Batch 4:  Loss:    13.1644 Validation Accuracy: 0.360600
    Epoch 30, CIFAR-10 Batch 5:  Loss:    16.4334 Validation Accuracy: 0.403800
    Epoch 31, CIFAR-10 Batch 1:  Loss:    10.7321 Validation Accuracy: 0.394600
    Epoch 31, CIFAR-10 Batch 2:  Loss:     2.1409 Validation Accuracy: 0.381800
    Epoch 31, CIFAR-10 Batch 3:  Loss:     3.8962 Validation Accuracy: 0.426600
    Epoch 31, CIFAR-10 Batch 4:  Loss:     2.3655 Validation Accuracy: 0.406400
    Epoch 31, CIFAR-10 Batch 5:  Loss:    20.6115 Validation Accuracy: 0.387400
    Epoch 32, CIFAR-10 Batch 1:  Loss:     7.4380 Validation Accuracy: 0.412400
    Epoch 32, CIFAR-10 Batch 2:  Loss:     2.9801 Validation Accuracy: 0.382000
    Epoch 32, CIFAR-10 Batch 3:  Loss:     9.5711 Validation Accuracy: 0.385200
    Epoch 32, CIFAR-10 Batch 4:  Loss:     1.8781 Validation Accuracy: 0.402200
    Epoch 32, CIFAR-10 Batch 5:  Loss:    18.4391 Validation Accuracy: 0.366000
    Epoch 33, CIFAR-10 Batch 1:  Loss:     7.0856 Validation Accuracy: 0.370600
    Epoch 33, CIFAR-10 Batch 2:  Loss:     5.2642 Validation Accuracy: 0.381600
    Epoch 33, CIFAR-10 Batch 3:  Loss:     7.1206 Validation Accuracy: 0.400600
    Epoch 33, CIFAR-10 Batch 4:  Loss:     3.0127 Validation Accuracy: 0.414000
    Epoch 33, CIFAR-10 Batch 5:  Loss:    16.7639 Validation Accuracy: 0.378400
    Epoch 34, CIFAR-10 Batch 1:  Loss:     4.5147 Validation Accuracy: 0.397000
    Epoch 34, CIFAR-10 Batch 2:  Loss:     2.2726 Validation Accuracy: 0.417000
    Epoch 34, CIFAR-10 Batch 3:  Loss:     3.2915 Validation Accuracy: 0.417400
    Epoch 34, CIFAR-10 Batch 4:  Loss:     3.1359 Validation Accuracy: 0.395800
    Epoch 34, CIFAR-10 Batch 5:  Loss:     3.4998 Validation Accuracy: 0.417400
    Epoch 35, CIFAR-10 Batch 1:  Loss:     2.0133 Validation Accuracy: 0.406000
    Epoch 35, CIFAR-10 Batch 2:  Loss:     4.7589 Validation Accuracy: 0.388800
    Epoch 35, CIFAR-10 Batch 3:  Loss:     1.3669 Validation Accuracy: 0.424200
    Epoch 35, CIFAR-10 Batch 4:  Loss:     1.5403 Validation Accuracy: 0.436000
    Epoch 35, CIFAR-10 Batch 5:  Loss:     2.9552 Validation Accuracy: 0.405400
    Epoch 36, CIFAR-10 Batch 1:  Loss:     2.1330 Validation Accuracy: 0.415600
    Epoch 36, CIFAR-10 Batch 2:  Loss:     2.7006 Validation Accuracy: 0.411000
    Epoch 36, CIFAR-10 Batch 3:  Loss:     2.8373 Validation Accuracy: 0.429200
    Epoch 36, CIFAR-10 Batch 4:  Loss:     1.9352 Validation Accuracy: 0.402600
    Epoch 36, CIFAR-10 Batch 5:  Loss:     2.4828 Validation Accuracy: 0.415800
    Epoch 37, CIFAR-10 Batch 1:  Loss:     2.2275 Validation Accuracy: 0.419600
    Epoch 37, CIFAR-10 Batch 2:  Loss:     2.4435 Validation Accuracy: 0.411800
    Epoch 37, CIFAR-10 Batch 3:  Loss:     2.9398 Validation Accuracy: 0.397600
    Epoch 37, CIFAR-10 Batch 4:  Loss:     1.8898 Validation Accuracy: 0.419400
    Epoch 37, CIFAR-10 Batch 5:  Loss:     1.8699 Validation Accuracy: 0.415600
    Epoch 38, CIFAR-10 Batch 1:  Loss:     1.4871 Validation Accuracy: 0.402200
    Epoch 38, CIFAR-10 Batch 2:  Loss:     2.6722 Validation Accuracy: 0.411000
    Epoch 38, CIFAR-10 Batch 3:  Loss:     1.7740 Validation Accuracy: 0.416800
    Epoch 38, CIFAR-10 Batch 4:  Loss:     1.4559 Validation Accuracy: 0.428800
    Epoch 38, CIFAR-10 Batch 5:  Loss:     1.6338 Validation Accuracy: 0.399200
    Epoch 39, CIFAR-10 Batch 1:  Loss:     1.6085 Validation Accuracy: 0.408000
    Epoch 39, CIFAR-10 Batch 2:  Loss:     1.9346 Validation Accuracy: 0.383200
    Epoch 39, CIFAR-10 Batch 3:  Loss:     1.5264 Validation Accuracy: 0.408800
    Epoch 39, CIFAR-10 Batch 4:  Loss:     1.3629 Validation Accuracy: 0.418000
    Epoch 39, CIFAR-10 Batch 5:  Loss:     1.8851 Validation Accuracy: 0.401600
    Epoch 40, CIFAR-10 Batch 1:  Loss:     1.4725 Validation Accuracy: 0.410000
    Epoch 40, CIFAR-10 Batch 2:  Loss:     2.1246 Validation Accuracy: 0.421600
    Epoch 40, CIFAR-10 Batch 3:  Loss:     1.6530 Validation Accuracy: 0.426400
    Epoch 40, CIFAR-10 Batch 4:  Loss:     1.4364 Validation Accuracy: 0.409200
    Epoch 40, CIFAR-10 Batch 5:  Loss:     1.4514 Validation Accuracy: 0.421000
    Epoch 41, CIFAR-10 Batch 1:  Loss:     1.3676 Validation Accuracy: 0.415400
    Epoch 41, CIFAR-10 Batch 2:  Loss:     1.8384 Validation Accuracy: 0.433600
    Epoch 41, CIFAR-10 Batch 3:  Loss:     1.1708 Validation Accuracy: 0.423200
    Epoch 41, CIFAR-10 Batch 4:  Loss:     1.4097 Validation Accuracy: 0.428400
    Epoch 41, CIFAR-10 Batch 5:  Loss:     1.4944 Validation Accuracy: 0.409400
    Epoch 42, CIFAR-10 Batch 1:  Loss:     1.3654 Validation Accuracy: 0.418800
    Epoch 42, CIFAR-10 Batch 2:  Loss:     1.5845 Validation Accuracy: 0.432400
    Epoch 42, CIFAR-10 Batch 3:  Loss:     1.6016 Validation Accuracy: 0.423200
    Epoch 42, CIFAR-10 Batch 4:  Loss:     1.3976 Validation Accuracy: 0.429400
    Epoch 42, CIFAR-10 Batch 5:  Loss:     1.4118 Validation Accuracy: 0.418600
    Epoch 43, CIFAR-10 Batch 1:  Loss:     1.4271 Validation Accuracy: 0.431800
    Epoch 43, CIFAR-10 Batch 2:  Loss:     1.4799 Validation Accuracy: 0.429800
    Epoch 43, CIFAR-10 Batch 3:  Loss:     1.2064 Validation Accuracy: 0.450400
    Epoch 43, CIFAR-10 Batch 4:  Loss:     1.4172 Validation Accuracy: 0.427600
    Epoch 43, CIFAR-10 Batch 5:  Loss:     1.3755 Validation Accuracy: 0.436600
    Epoch 44, CIFAR-10 Batch 1:  Loss:     1.2831 Validation Accuracy: 0.443800
    Epoch 44, CIFAR-10 Batch 2:  Loss:     1.4377 Validation Accuracy: 0.443400
    Epoch 44, CIFAR-10 Batch 3:  Loss:     1.2520 Validation Accuracy: 0.457000
    Epoch 44, CIFAR-10 Batch 4:  Loss:     1.4051 Validation Accuracy: 0.445800
    Epoch 44, CIFAR-10 Batch 5:  Loss:     1.3535 Validation Accuracy: 0.445400
    Epoch 45, CIFAR-10 Batch 1:  Loss:     1.1544 Validation Accuracy: 0.466600
    Epoch 45, CIFAR-10 Batch 2:  Loss:     1.2718 Validation Accuracy: 0.462200
    Epoch 45, CIFAR-10 Batch 3:  Loss:     1.1746 Validation Accuracy: 0.472000
    Epoch 45, CIFAR-10 Batch 4:  Loss:     1.3968 Validation Accuracy: 0.471600
    Epoch 45, CIFAR-10 Batch 5:  Loss:     1.2429 Validation Accuracy: 0.461200
    Epoch 46, CIFAR-10 Batch 1:  Loss:     1.1852 Validation Accuracy: 0.470800
    Epoch 46, CIFAR-10 Batch 2:  Loss:     1.3356 Validation Accuracy: 0.468600
    Epoch 46, CIFAR-10 Batch 3:  Loss:     1.1051 Validation Accuracy: 0.471600
    Epoch 46, CIFAR-10 Batch 4:  Loss:     1.1988 Validation Accuracy: 0.481400
    Epoch 46, CIFAR-10 Batch 5:  Loss:     1.1450 Validation Accuracy: 0.480000
    Epoch 47, CIFAR-10 Batch 1:  Loss:     1.1667 Validation Accuracy: 0.481600
    Epoch 47, CIFAR-10 Batch 2:  Loss:     1.2631 Validation Accuracy: 0.481000
    Epoch 47, CIFAR-10 Batch 3:  Loss:     1.0665 Validation Accuracy: 0.480400
    Epoch 47, CIFAR-10 Batch 4:  Loss:     1.2271 Validation Accuracy: 0.487600
    Epoch 47, CIFAR-10 Batch 5:  Loss:     1.0408 Validation Accuracy: 0.483400
    Epoch 48, CIFAR-10 Batch 1:  Loss:     1.1269 Validation Accuracy: 0.484600
    Epoch 48, CIFAR-10 Batch 2:  Loss:     1.3022 Validation Accuracy: 0.482400
    Epoch 48, CIFAR-10 Batch 3:  Loss:     0.9948 Validation Accuracy: 0.494200
    Epoch 48, CIFAR-10 Batch 4:  Loss:     1.1456 Validation Accuracy: 0.497400
    Epoch 48, CIFAR-10 Batch 5:  Loss:     1.0972 Validation Accuracy: 0.498600
    Epoch 49, CIFAR-10 Batch 1:  Loss:     1.0373 Validation Accuracy: 0.491200
    Epoch 49, CIFAR-10 Batch 2:  Loss:     1.1991 Validation Accuracy: 0.490800
    Epoch 49, CIFAR-10 Batch 3:  Loss:     0.9500 Validation Accuracy: 0.500000
    Epoch 49, CIFAR-10 Batch 4:  Loss:     1.1221 Validation Accuracy: 0.511200
    Epoch 49, CIFAR-10 Batch 5:  Loss:     1.0442 Validation Accuracy: 0.512200
    Epoch 50, CIFAR-10 Batch 1:  Loss:     1.1392 Validation Accuracy: 0.511200
    Epoch 50, CIFAR-10 Batch 2:  Loss:     1.1357 Validation Accuracy: 0.503600
    Epoch 50, CIFAR-10 Batch 3:  Loss:     0.9351 Validation Accuracy: 0.519800
    Epoch 50, CIFAR-10 Batch 4:  Loss:     1.0380 Validation Accuracy: 0.516400
    Epoch 50, CIFAR-10 Batch 5:  Loss:     1.0346 Validation Accuracy: 0.518600
    Epoch 51, CIFAR-10 Batch 1:  Loss:     1.0475 Validation Accuracy: 0.522000
    Epoch 51, CIFAR-10 Batch 2:  Loss:     1.1229 Validation Accuracy: 0.503600
    Epoch 51, CIFAR-10 Batch 3:  Loss:     0.9583 Validation Accuracy: 0.520600
    Epoch 51, CIFAR-10 Batch 4:  Loss:     0.9600 Validation Accuracy: 0.499800
    Epoch 51, CIFAR-10 Batch 5:  Loss:     1.0227 Validation Accuracy: 0.517000
    Epoch 52, CIFAR-10 Batch 1:  Loss:     1.0220 Validation Accuracy: 0.523400
    Epoch 52, CIFAR-10 Batch 2:  Loss:     1.0575 Validation Accuracy: 0.526400
    Epoch 52, CIFAR-10 Batch 3:  Loss:     0.8702 Validation Accuracy: 0.516600
    Epoch 52, CIFAR-10 Batch 4:  Loss:     0.9750 Validation Accuracy: 0.508600
    Epoch 52, CIFAR-10 Batch 5:  Loss:     0.9622 Validation Accuracy: 0.520800
    Epoch 53, CIFAR-10 Batch 1:  Loss:     0.9744 Validation Accuracy: 0.524600
    Epoch 53, CIFAR-10 Batch 2:  Loss:     0.9764 Validation Accuracy: 0.540000
    Epoch 53, CIFAR-10 Batch 3:  Loss:     0.8349 Validation Accuracy: 0.535600
    Epoch 53, CIFAR-10 Batch 4:  Loss:     0.8173 Validation Accuracy: 0.528200
    Epoch 53, CIFAR-10 Batch 5:  Loss:     0.9912 Validation Accuracy: 0.526400
    Epoch 54, CIFAR-10 Batch 1:  Loss:     0.9899 Validation Accuracy: 0.532800
    Epoch 54, CIFAR-10 Batch 2:  Loss:     0.9337 Validation Accuracy: 0.539400
    Epoch 54, CIFAR-10 Batch 3:  Loss:     0.7129 Validation Accuracy: 0.535800
    Epoch 54, CIFAR-10 Batch 4:  Loss:     0.8685 Validation Accuracy: 0.515600
    Epoch 54, CIFAR-10 Batch 5:  Loss:     0.8911 Validation Accuracy: 0.532600
    Epoch 55, CIFAR-10 Batch 1:  Loss:     0.9013 Validation Accuracy: 0.532000
    Epoch 55, CIFAR-10 Batch 2:  Loss:     0.8918 Validation Accuracy: 0.529800
    Epoch 55, CIFAR-10 Batch 3:  Loss:     0.7429 Validation Accuracy: 0.529400
    Epoch 55, CIFAR-10 Batch 4:  Loss:     0.7959 Validation Accuracy: 0.530200
    Epoch 55, CIFAR-10 Batch 5:  Loss:     0.8998 Validation Accuracy: 0.531000
    Epoch 56, CIFAR-10 Batch 1:  Loss:     0.9844 Validation Accuracy: 0.535600
    Epoch 56, CIFAR-10 Batch 2:  Loss:     0.8921 Validation Accuracy: 0.547200
    Epoch 56, CIFAR-10 Batch 3:  Loss:     0.7561 Validation Accuracy: 0.540200
    Epoch 56, CIFAR-10 Batch 4:  Loss:     0.7503 Validation Accuracy: 0.531600
    Epoch 56, CIFAR-10 Batch 5:  Loss:     0.8569 Validation Accuracy: 0.536200
    Epoch 57, CIFAR-10 Batch 1:  Loss:     0.8917 Validation Accuracy: 0.544600
    Epoch 57, CIFAR-10 Batch 2:  Loss:     0.8135 Validation Accuracy: 0.542000
    Epoch 57, CIFAR-10 Batch 3:  Loss:     0.6629 Validation Accuracy: 0.550800
    Epoch 57, CIFAR-10 Batch 4:  Loss:     0.7406 Validation Accuracy: 0.536600
    Epoch 57, CIFAR-10 Batch 5:  Loss:     0.7190 Validation Accuracy: 0.542800
    Epoch 58, CIFAR-10 Batch 1:  Loss:     0.8392 Validation Accuracy: 0.540600
    Epoch 58, CIFAR-10 Batch 2:  Loss:     0.7648 Validation Accuracy: 0.546200
    Epoch 58, CIFAR-10 Batch 3:  Loss:     0.6080 Validation Accuracy: 0.563000
    Epoch 58, CIFAR-10 Batch 4:  Loss:     0.6938 Validation Accuracy: 0.529600
    Epoch 58, CIFAR-10 Batch 5:  Loss:     0.8303 Validation Accuracy: 0.538200
    Epoch 59, CIFAR-10 Batch 1:  Loss:     0.8315 Validation Accuracy: 0.524000
    Epoch 59, CIFAR-10 Batch 2:  Loss:     0.7848 Validation Accuracy: 0.557000
    Epoch 59, CIFAR-10 Batch 3:  Loss:     0.5884 Validation Accuracy: 0.563000
    Epoch 59, CIFAR-10 Batch 4:  Loss:     0.5944 Validation Accuracy: 0.555600
    Epoch 59, CIFAR-10 Batch 5:  Loss:     0.6835 Validation Accuracy: 0.546000
    Epoch 60, CIFAR-10 Batch 1:  Loss:     0.6293 Validation Accuracy: 0.549200
    Epoch 60, CIFAR-10 Batch 2:  Loss:     0.6965 Validation Accuracy: 0.545600
    Epoch 60, CIFAR-10 Batch 3:  Loss:     0.5692 Validation Accuracy: 0.559000
    Epoch 60, CIFAR-10 Batch 4:  Loss:     0.5376 Validation Accuracy: 0.551400
    Epoch 60, CIFAR-10 Batch 5:  Loss:     0.7314 Validation Accuracy: 0.572000
    Epoch 61, CIFAR-10 Batch 1:  Loss:     0.8290 Validation Accuracy: 0.551600
    Epoch 61, CIFAR-10 Batch 2:  Loss:     0.6755 Validation Accuracy: 0.559600
    Epoch 61, CIFAR-10 Batch 3:  Loss:     0.4261 Validation Accuracy: 0.567000
    Epoch 61, CIFAR-10 Batch 4:  Loss:     0.5634 Validation Accuracy: 0.566600
    Epoch 61, CIFAR-10 Batch 5:  Loss:     0.7274 Validation Accuracy: 0.560800
    Epoch 62, CIFAR-10 Batch 1:  Loss:     0.7050 Validation Accuracy: 0.552600
    Epoch 62, CIFAR-10 Batch 2:  Loss:     0.6329 Validation Accuracy: 0.568000
    Epoch 62, CIFAR-10 Batch 3:  Loss:     0.4771 Validation Accuracy: 0.559200
    Epoch 62, CIFAR-10 Batch 4:  Loss:     0.5255 Validation Accuracy: 0.563800
    Epoch 62, CIFAR-10 Batch 5:  Loss:     0.6772 Validation Accuracy: 0.553400
    Epoch 63, CIFAR-10 Batch 1:  Loss:     0.6256 Validation Accuracy: 0.545000
    Epoch 63, CIFAR-10 Batch 2:  Loss:     0.6039 Validation Accuracy: 0.562200
    Epoch 63, CIFAR-10 Batch 3:  Loss:     0.4650 Validation Accuracy: 0.569400
    Epoch 63, CIFAR-10 Batch 4:  Loss:     0.4967 Validation Accuracy: 0.571400
    Epoch 63, CIFAR-10 Batch 5:  Loss:     0.6017 Validation Accuracy: 0.567600
    Epoch 64, CIFAR-10 Batch 1:  Loss:     0.5421 Validation Accuracy: 0.560800
    Epoch 64, CIFAR-10 Batch 2:  Loss:     0.5449 Validation Accuracy: 0.563000
    Epoch 64, CIFAR-10 Batch 3:  Loss:     0.3756 Validation Accuracy: 0.576200
    Epoch 64, CIFAR-10 Batch 4:  Loss:     0.4586 Validation Accuracy: 0.548400
    Epoch 64, CIFAR-10 Batch 5:  Loss:     0.5192 Validation Accuracy: 0.561200
    Epoch 65, CIFAR-10 Batch 1:  Loss:     0.4664 Validation Accuracy: 0.566000
    Epoch 65, CIFAR-10 Batch 2:  Loss:     0.6304 Validation Accuracy: 0.575800
    Epoch 65, CIFAR-10 Batch 3:  Loss:     0.3281 Validation Accuracy: 0.576400
    Epoch 65, CIFAR-10 Batch 4:  Loss:     0.3852 Validation Accuracy: 0.572800
    Epoch 65, CIFAR-10 Batch 5:  Loss:     0.4912 Validation Accuracy: 0.577200
    Epoch 66, CIFAR-10 Batch 1:  Loss:     0.5030 Validation Accuracy: 0.578600
    Epoch 66, CIFAR-10 Batch 2:  Loss:     0.5048 Validation Accuracy: 0.587600
    Epoch 66, CIFAR-10 Batch 3:  Loss:     0.2519 Validation Accuracy: 0.578000
    Epoch 66, CIFAR-10 Batch 4:  Loss:     0.3651 Validation Accuracy: 0.568800
    Epoch 66, CIFAR-10 Batch 5:  Loss:     0.5841 Validation Accuracy: 0.583400
    Epoch 67, CIFAR-10 Batch 1:  Loss:     0.5122 Validation Accuracy: 0.571000
    Epoch 67, CIFAR-10 Batch 2:  Loss:     0.4228 Validation Accuracy: 0.569400
    Epoch 67, CIFAR-10 Batch 3:  Loss:     0.1965 Validation Accuracy: 0.578600
    Epoch 67, CIFAR-10 Batch 4:  Loss:     0.3656 Validation Accuracy: 0.570400
    Epoch 67, CIFAR-10 Batch 5:  Loss:     0.4423 Validation Accuracy: 0.582000
    Epoch 68, CIFAR-10 Batch 1:  Loss:     0.4400 Validation Accuracy: 0.580400
    Epoch 68, CIFAR-10 Batch 2:  Loss:     0.3845 Validation Accuracy: 0.578400
    Epoch 68, CIFAR-10 Batch 3:  Loss:     0.2577 Validation Accuracy: 0.582000
    Epoch 68, CIFAR-10 Batch 4:  Loss:     0.3150 Validation Accuracy: 0.583200
    Epoch 68, CIFAR-10 Batch 5:  Loss:     0.3984 Validation Accuracy: 0.579400
    Epoch 69, CIFAR-10 Batch 1:  Loss:     0.3684 Validation Accuracy: 0.576400
    Epoch 69, CIFAR-10 Batch 2:  Loss:     0.4572 Validation Accuracy: 0.570200
    Epoch 69, CIFAR-10 Batch 3:  Loss:     0.2143 Validation Accuracy: 0.578200
    Epoch 69, CIFAR-10 Batch 4:  Loss:     0.3276 Validation Accuracy: 0.567000
    Epoch 69, CIFAR-10 Batch 5:  Loss:     0.3178 Validation Accuracy: 0.583200
    Epoch 70, CIFAR-10 Batch 1:  Loss:     0.3636 Validation Accuracy: 0.570400
    Epoch 70, CIFAR-10 Batch 2:  Loss:     0.3918 Validation Accuracy: 0.580400
    Epoch 70, CIFAR-10 Batch 3:  Loss:     0.2124 Validation Accuracy: 0.584200
    Epoch 70, CIFAR-10 Batch 4:  Loss:     0.2869 Validation Accuracy: 0.571200
    Epoch 70, CIFAR-10 Batch 5:  Loss:     0.3064 Validation Accuracy: 0.591600
    Epoch 71, CIFAR-10 Batch 1:  Loss:     0.4301 Validation Accuracy: 0.579200
    Epoch 71, CIFAR-10 Batch 2:  Loss:     0.3656 Validation Accuracy: 0.593000
    Epoch 71, CIFAR-10 Batch 3:  Loss:     0.2231 Validation Accuracy: 0.583200
    Epoch 71, CIFAR-10 Batch 4:  Loss:     0.3011 Validation Accuracy: 0.579000
    Epoch 71, CIFAR-10 Batch 5:  Loss:     0.3104 Validation Accuracy: 0.593400
    Epoch 72, CIFAR-10 Batch 1:  Loss:     0.4047 Validation Accuracy: 0.574400
    Epoch 72, CIFAR-10 Batch 2:  Loss:     0.2847 Validation Accuracy: 0.588000
    Epoch 72, CIFAR-10 Batch 3:  Loss:     0.2156 Validation Accuracy: 0.593800
    Epoch 72, CIFAR-10 Batch 4:  Loss:     0.3803 Validation Accuracy: 0.580600
    Epoch 72, CIFAR-10 Batch 5:  Loss:     0.2872 Validation Accuracy: 0.594200
    Epoch 73, CIFAR-10 Batch 1:  Loss:     0.4040 Validation Accuracy: 0.577600
    Epoch 73, CIFAR-10 Batch 2:  Loss:     0.3696 Validation Accuracy: 0.587800
    Epoch 73, CIFAR-10 Batch 3:  Loss:     0.1288 Validation Accuracy: 0.592200
    Epoch 73, CIFAR-10 Batch 4:  Loss:     0.2646 Validation Accuracy: 0.588200
    Epoch 73, CIFAR-10 Batch 5:  Loss:     0.3431 Validation Accuracy: 0.587800
    Epoch 74, CIFAR-10 Batch 1:  Loss:     0.3185 Validation Accuracy: 0.579000
    Epoch 74, CIFAR-10 Batch 2:  Loss:     0.3603 Validation Accuracy: 0.595000
    Epoch 74, CIFAR-10 Batch 3:  Loss:     0.1225 Validation Accuracy: 0.600600
    Epoch 74, CIFAR-10 Batch 4:  Loss:     0.2135 Validation Accuracy: 0.585200
    Epoch 74, CIFAR-10 Batch 5:  Loss:     0.5392 Validation Accuracy: 0.594600
    Epoch 75, CIFAR-10 Batch 1:  Loss:     0.3500 Validation Accuracy: 0.592200
    Epoch 75, CIFAR-10 Batch 2:  Loss:     0.3027 Validation Accuracy: 0.605600
    Epoch 75, CIFAR-10 Batch 3:  Loss:     0.1342 Validation Accuracy: 0.598000
    Epoch 75, CIFAR-10 Batch 4:  Loss:     0.1939 Validation Accuracy: 0.585600
    Epoch 75, CIFAR-10 Batch 5:  Loss:     0.2086 Validation Accuracy: 0.604600
    Epoch 76, CIFAR-10 Batch 1:  Loss:     0.2576 Validation Accuracy: 0.597000
    Epoch 76, CIFAR-10 Batch 2:  Loss:     0.2917 Validation Accuracy: 0.600600
    Epoch 76, CIFAR-10 Batch 3:  Loss:     0.2251 Validation Accuracy: 0.599200
    Epoch 76, CIFAR-10 Batch 4:  Loss:     0.2082 Validation Accuracy: 0.588000
    Epoch 76, CIFAR-10 Batch 5:  Loss:     0.1988 Validation Accuracy: 0.601600
    Epoch 77, CIFAR-10 Batch 1:  Loss:     0.3297 Validation Accuracy: 0.594600
    Epoch 77, CIFAR-10 Batch 2:  Loss:     0.2843 Validation Accuracy: 0.599600
    Epoch 77, CIFAR-10 Batch 3:  Loss:     0.1151 Validation Accuracy: 0.596800
    Epoch 77, CIFAR-10 Batch 4:  Loss:     0.2278 Validation Accuracy: 0.595400
    Epoch 77, CIFAR-10 Batch 5:  Loss:     0.2401 Validation Accuracy: 0.605200
    Epoch 78, CIFAR-10 Batch 1:  Loss:     0.2709 Validation Accuracy: 0.593800
    Epoch 78, CIFAR-10 Batch 2:  Loss:     0.2860 Validation Accuracy: 0.594600
    Epoch 78, CIFAR-10 Batch 3:  Loss:     0.1715 Validation Accuracy: 0.600400
    Epoch 78, CIFAR-10 Batch 4:  Loss:     0.1669 Validation Accuracy: 0.597800
    Epoch 78, CIFAR-10 Batch 5:  Loss:     0.2037 Validation Accuracy: 0.607000
    Epoch 79, CIFAR-10 Batch 1:  Loss:     0.2310 Validation Accuracy: 0.595400
    Epoch 79, CIFAR-10 Batch 2:  Loss:     0.2773 Validation Accuracy: 0.597400
    Epoch 79, CIFAR-10 Batch 3:  Loss:     0.1565 Validation Accuracy: 0.603600
    Epoch 79, CIFAR-10 Batch 4:  Loss:     0.1852 Validation Accuracy: 0.594800
    Epoch 79, CIFAR-10 Batch 5:  Loss:     0.1797 Validation Accuracy: 0.601600
    Epoch 80, CIFAR-10 Batch 1:  Loss:     0.2571 Validation Accuracy: 0.602600
    Epoch 80, CIFAR-10 Batch 2:  Loss:     0.2014 Validation Accuracy: 0.596200
    Epoch 80, CIFAR-10 Batch 3:  Loss:     0.0981 Validation Accuracy: 0.592000
    Epoch 80, CIFAR-10 Batch 4:  Loss:     0.1307 Validation Accuracy: 0.592800
    Epoch 80, CIFAR-10 Batch 5:  Loss:     0.1483 Validation Accuracy: 0.604800
    Epoch 81, CIFAR-10 Batch 1:  Loss:     0.2314 Validation Accuracy: 0.606800
    Epoch 81, CIFAR-10 Batch 2:  Loss:     0.2601 Validation Accuracy: 0.590600
    Epoch 81, CIFAR-10 Batch 3:  Loss:     0.1154 Validation Accuracy: 0.600800
    Epoch 81, CIFAR-10 Batch 4:  Loss:     0.1432 Validation Accuracy: 0.590200
    Epoch 81, CIFAR-10 Batch 5:  Loss:     0.1627 Validation Accuracy: 0.603600
    Epoch 82, CIFAR-10 Batch 1:  Loss:     0.2581 Validation Accuracy: 0.607400
    Epoch 82, CIFAR-10 Batch 2:  Loss:     0.2573 Validation Accuracy: 0.610600
    Epoch 82, CIFAR-10 Batch 3:  Loss:     0.0843 Validation Accuracy: 0.609200
    Epoch 82, CIFAR-10 Batch 4:  Loss:     0.1512 Validation Accuracy: 0.608600
    Epoch 82, CIFAR-10 Batch 5:  Loss:     0.1469 Validation Accuracy: 0.604000
    Epoch 83, CIFAR-10 Batch 1:  Loss:     0.2147 Validation Accuracy: 0.608200
    Epoch 83, CIFAR-10 Batch 2:  Loss:     0.1974 Validation Accuracy: 0.597800
    Epoch 83, CIFAR-10 Batch 3:  Loss:     0.0872 Validation Accuracy: 0.607200
    Epoch 83, CIFAR-10 Batch 4:  Loss:     0.1328 Validation Accuracy: 0.601000
    Epoch 83, CIFAR-10 Batch 5:  Loss:     0.1604 Validation Accuracy: 0.609600
    Epoch 84, CIFAR-10 Batch 1:  Loss:     0.2203 Validation Accuracy: 0.617600
    Epoch 84, CIFAR-10 Batch 2:  Loss:     0.2327 Validation Accuracy: 0.595400
    Epoch 84, CIFAR-10 Batch 3:  Loss:     0.1251 Validation Accuracy: 0.604600
    Epoch 84, CIFAR-10 Batch 4:  Loss:     0.1207 Validation Accuracy: 0.610600
    Epoch 84, CIFAR-10 Batch 5:  Loss:     0.1181 Validation Accuracy: 0.607400
    Epoch 85, CIFAR-10 Batch 1:  Loss:     0.1838 Validation Accuracy: 0.598000
    Epoch 85, CIFAR-10 Batch 2:  Loss:     0.1811 Validation Accuracy: 0.596400
    Epoch 85, CIFAR-10 Batch 3:  Loss:     0.0661 Validation Accuracy: 0.604800
    Epoch 85, CIFAR-10 Batch 4:  Loss:     0.1460 Validation Accuracy: 0.604400
    Epoch 85, CIFAR-10 Batch 5:  Loss:     0.1366 Validation Accuracy: 0.598200
    Epoch 86, CIFAR-10 Batch 1:  Loss:     0.2285 Validation Accuracy: 0.618800
    Epoch 86, CIFAR-10 Batch 2:  Loss:     0.1749 Validation Accuracy: 0.605800
    Epoch 86, CIFAR-10 Batch 3:  Loss:     0.1407 Validation Accuracy: 0.601600
    Epoch 86, CIFAR-10 Batch 4:  Loss:     0.1259 Validation Accuracy: 0.610800
    Epoch 86, CIFAR-10 Batch 5:  Loss:     0.0954 Validation Accuracy: 0.612200
    Epoch 87, CIFAR-10 Batch 1:  Loss:     0.1844 Validation Accuracy: 0.594800
    Epoch 87, CIFAR-10 Batch 2:  Loss:     0.2181 Validation Accuracy: 0.602400
    Epoch 87, CIFAR-10 Batch 3:  Loss:     0.0618 Validation Accuracy: 0.609200
    Epoch 87, CIFAR-10 Batch 4:  Loss:     0.1106 Validation Accuracy: 0.610600
    Epoch 87, CIFAR-10 Batch 5:  Loss:     0.1435 Validation Accuracy: 0.608600
    Epoch 88, CIFAR-10 Batch 1:  Loss:     0.1788 Validation Accuracy: 0.621400
    Epoch 88, CIFAR-10 Batch 2:  Loss:     0.1249 Validation Accuracy: 0.615200
    Epoch 88, CIFAR-10 Batch 3:  Loss:     0.0459 Validation Accuracy: 0.612000
    Epoch 88, CIFAR-10 Batch 4:  Loss:     0.0803 Validation Accuracy: 0.609000
    Epoch 88, CIFAR-10 Batch 5:  Loss:     0.1238 Validation Accuracy: 0.609800
    Epoch 89, CIFAR-10 Batch 1:  Loss:     0.1777 Validation Accuracy: 0.606800
    Epoch 89, CIFAR-10 Batch 2:  Loss:     0.1655 Validation Accuracy: 0.610400
    Epoch 89, CIFAR-10 Batch 3:  Loss:     0.0868 Validation Accuracy: 0.608600
    Epoch 89, CIFAR-10 Batch 4:  Loss:     0.1043 Validation Accuracy: 0.617200
    Epoch 89, CIFAR-10 Batch 5:  Loss:     0.2006 Validation Accuracy: 0.623200
    Epoch 90, CIFAR-10 Batch 1:  Loss:     0.2079 Validation Accuracy: 0.616200
    Epoch 90, CIFAR-10 Batch 2:  Loss:     0.1512 Validation Accuracy: 0.613400
    Epoch 90, CIFAR-10 Batch 3:  Loss:     0.0665 Validation Accuracy: 0.611200
    Epoch 90, CIFAR-10 Batch 4:  Loss:     0.1129 Validation Accuracy: 0.615800
    Epoch 90, CIFAR-10 Batch 5:  Loss:     0.0957 Validation Accuracy: 0.607000
    Epoch 91, CIFAR-10 Batch 1:  Loss:     0.1799 Validation Accuracy: 0.617400
    Epoch 91, CIFAR-10 Batch 2:  Loss:     0.2186 Validation Accuracy: 0.617000
    Epoch 91, CIFAR-10 Batch 3:  Loss:     0.0765 Validation Accuracy: 0.623600
    Epoch 91, CIFAR-10 Batch 4:  Loss:     0.1115 Validation Accuracy: 0.625400
    Epoch 91, CIFAR-10 Batch 5:  Loss:     0.1113 Validation Accuracy: 0.612400
    Epoch 92, CIFAR-10 Batch 1:  Loss:     0.1837 Validation Accuracy: 0.617000
    Epoch 92, CIFAR-10 Batch 2:  Loss:     0.1432 Validation Accuracy: 0.623800
    Epoch 92, CIFAR-10 Batch 3:  Loss:     0.0591 Validation Accuracy: 0.628000
    Epoch 92, CIFAR-10 Batch 4:  Loss:     0.1126 Validation Accuracy: 0.612400
    Epoch 92, CIFAR-10 Batch 5:  Loss:     0.1000 Validation Accuracy: 0.617800
    Epoch 93, CIFAR-10 Batch 1:  Loss:     0.1820 Validation Accuracy: 0.618000
    Epoch 93, CIFAR-10 Batch 2:  Loss:     0.1313 Validation Accuracy: 0.624400
    Epoch 93, CIFAR-10 Batch 3:  Loss:     0.0644 Validation Accuracy: 0.625600
    Epoch 93, CIFAR-10 Batch 4:  Loss:     0.1025 Validation Accuracy: 0.623600
    Epoch 93, CIFAR-10 Batch 5:  Loss:     0.0775 Validation Accuracy: 0.618000
    Epoch 94, CIFAR-10 Batch 1:  Loss:     0.1340 Validation Accuracy: 0.624000
    Epoch 94, CIFAR-10 Batch 2:  Loss:     0.1474 Validation Accuracy: 0.624000
    Epoch 94, CIFAR-10 Batch 3:  Loss:     0.0743 Validation Accuracy: 0.621600
    Epoch 94, CIFAR-10 Batch 4:  Loss:     0.0729 Validation Accuracy: 0.621400
    Epoch 94, CIFAR-10 Batch 5:  Loss:     0.0960 Validation Accuracy: 0.617600
    Epoch 95, CIFAR-10 Batch 1:  Loss:     0.1553 Validation Accuracy: 0.621400
    Epoch 95, CIFAR-10 Batch 2:  Loss:     0.1682 Validation Accuracy: 0.620600
    Epoch 95, CIFAR-10 Batch 3:  Loss:     0.0511 Validation Accuracy: 0.617000
    Epoch 95, CIFAR-10 Batch 4:  Loss:     0.0888 Validation Accuracy: 0.623200
    Epoch 95, CIFAR-10 Batch 5:  Loss:     0.0836 Validation Accuracy: 0.624600
    Epoch 96, CIFAR-10 Batch 1:  Loss:     0.1818 Validation Accuracy: 0.621200
    Epoch 96, CIFAR-10 Batch 2:  Loss:     0.1531 Validation Accuracy: 0.615000
    Epoch 96, CIFAR-10 Batch 3:  Loss:     0.0335 Validation Accuracy: 0.623200
    Epoch 96, CIFAR-10 Batch 4:  Loss:     0.0813 Validation Accuracy: 0.617800
    Epoch 96, CIFAR-10 Batch 5:  Loss:     0.1049 Validation Accuracy: 0.617200
    Epoch 97, CIFAR-10 Batch 1:  Loss:     0.1292 Validation Accuracy: 0.624200
    Epoch 97, CIFAR-10 Batch 2:  Loss:     0.0948 Validation Accuracy: 0.610800
    Epoch 97, CIFAR-10 Batch 3:  Loss:     0.0527 Validation Accuracy: 0.608400
    Epoch 97, CIFAR-10 Batch 4:  Loss:     0.1012 Validation Accuracy: 0.612800
    Epoch 97, CIFAR-10 Batch 5:  Loss:     0.1022 Validation Accuracy: 0.613800
    Epoch 98, CIFAR-10 Batch 1:  Loss:     0.1286 Validation Accuracy: 0.621400
    Epoch 98, CIFAR-10 Batch 2:  Loss:     0.1107 Validation Accuracy: 0.620400
    Epoch 98, CIFAR-10 Batch 3:  Loss:     0.0584 Validation Accuracy: 0.609400
    Epoch 98, CIFAR-10 Batch 4:  Loss:     0.0816 Validation Accuracy: 0.618400
    Epoch 98, CIFAR-10 Batch 5:  Loss:     0.0879 Validation Accuracy: 0.616400
    Epoch 99, CIFAR-10 Batch 1:  Loss:     0.1301 Validation Accuracy: 0.620200
    Epoch 99, CIFAR-10 Batch 2:  Loss:     0.1249 Validation Accuracy: 0.627600
    Epoch 99, CIFAR-10 Batch 3:  Loss:     0.0446 Validation Accuracy: 0.613800
    Epoch 99, CIFAR-10 Batch 4:  Loss:     0.0664 Validation Accuracy: 0.624000
    Epoch 99, CIFAR-10 Batch 5:  Loss:     0.2240 Validation Accuracy: 0.616800
    Epoch 100, CIFAR-10 Batch 1:  Loss:     0.1259 Validation Accuracy: 0.627200
    Epoch 100, CIFAR-10 Batch 2:  Loss:     0.1871 Validation Accuracy: 0.619800
    Epoch 100, CIFAR-10 Batch 3:  Loss:     0.0455 Validation Accuracy: 0.628200
    Epoch 100, CIFAR-10 Batch 4:  Loss:     0.0618 Validation Accuracy: 0.636600
    Epoch 100, CIFAR-10 Batch 5:  Loss:     0.0568 Validation Accuracy: 0.632000


## Test Model
Tune Hyperparameters and Test your model against the test dataset, until accuracy is greater than 50%. 


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
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
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
```

    Testing Accuracy: 0.6241099683544303
    



![png](output_32_1.png)

