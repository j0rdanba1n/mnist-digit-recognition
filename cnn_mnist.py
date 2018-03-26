from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Application logic will be added here

if __name__ == "__main__":
    tf.app.run()

def cnn_model_fn (features, labels, mode):
    """Model function for Convolutional Neural Networks"""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d
    (
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    #Convolutional Layer #2
    conv2 = tf.layers.conv2d
    (
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)

    # Reshape pool2 into [batch_size, features] tensor
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

    # Dense Layer
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)

    # Dropout
    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = (mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs = dropout, units = 10)

    # Extract predictions
    predictions =
    {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    # If in TRAIN mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpac(mode = mode, predictions = predictions)

    # Calculate loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)

    # Configure the training optimizer (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Calculate evaluation metrics (for EVAL mode)
    eval_metric_ops =
    {
        "accuracy" : tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)
