import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.training import moving_averages


def _create_variable(name, shape, initializer):
    var = tf.compat.v1.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, wd=None):
    # Get the number of input and output parameters
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))

    # He et al. 2015 - http://arxiv.org/abs/1502.01852
    stddev = np.sqrt(2.0 / fan_in)
    initializer = tf.truncated_normal_initializer(stddev=stddev)

    # # Xavier
    # initializer = tf.contrib.layers.xavier_initializer()

    # Create or get the existing variable
    var = _create_variable(
        name,
        shape,
        initializer
    )

    # L2 weight decay
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.compat.v1.add_to_collection("losses", weight_decay)

    return var


def conv_1d(name, input_var, filter_shape, stride, padding="SAME", 
            bias=None, wd=None):
    with tf.compat.v1.variable_scope(name) as scope:
        # Trainable parameters
        kernel = variable_with_weight_decay(
            "weights",
            shape=filter_shape,
            wd=wd
        )

        # Convolution
        output_var = tf.nn.conv2d(
            input_var,
            kernel,
            [1, stride, 1, 1],
            padding=padding
        )

        # Bias
        if bias is not None:
            biases = _create_variable(
                "biases",
                [filter_shape[-1]],
                tf.constant_initializer(bias)
            )
            output_var = tf.nn.bias_add(output_var, biases)

        return output_var


def max_pool_1d(name, input_var, pool_size, stride, padding="SAME"):
    output_var = tf.nn.max_pool2d(
        input_var,
        ksize=[1, pool_size, 1, 1],
        strides=[1, stride, 1, 1],
        padding=padding,
        name=name
    )

    return output_var


def avg_pool_1d(name, input_var, pool_size, stride, padding="SAME"):
    output_var = tf.nn.avg_pool(
        input_var,
        ksize=[1, pool_size, 1, 1],
        strides=[1, stride, 1, 1],
        padding=padding,
        name=name
    )

    return output_var


def fc(name, input_var, n_hiddens, bias=None, wd=None):
    with tf.compat.v1.variable_scope(name) as scope:
        # Get input dimension
        input_dim = input_var.get_shape()[-1].value

        # Trainable parameters
        weights = variable_with_weight_decay(
            "weights",
            shape=[input_dim, n_hiddens],
            wd=wd
        )

        # Multiply weights
        output_var = tf.matmul(input_var, weights)

        # Bias
        if bias is not None:
            biases = _create_variable(
                "biases",
                [n_hiddens],
                tf.constant_initializer(bias)
            )
            output_var = tf.add(output_var, biases)

        return output_var


def leaky_relu(name, input_var, alpha=0.01):
    return tf.maximum(
        input_var, 
        alpha * input_var,
        name="leaky_relu"
    )


def batch_norm(name, input_var, is_train, decay=0.999, epsilon=1e-5):
    """Batch normalization on fully-connected or convolutional maps.
    Source: <http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow>
    """

    inputs_shape = input_var.get_shape()
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]

    with tf.compat.v1.variable_scope(name) as scope:
      beta = tf.compat.v1.get_variable(name="beta", shape=params_shape, 
                             initializer=tf.constant_initializer(0.0))
      gamma = tf.compat.v1.get_variable(name="gamma", shape=params_shape, 
                              initializer=tf.constant_initializer(1.0))
      batch_mean, batch_var = tf.nn.moments(input_var,
                                            axis,
                                            name="moments")
      ema = tf.train.ExponentialMovingAverage(decay=decay)

      def mean_var_with_update():
          ema_apply_op = ema.apply([batch_mean, batch_var])
          with tf.control_dependencies([ema_apply_op]):
              return tf.identity(batch_mean), tf.identity(batch_var)

      mean, var = tf.cond(
          is_train,
          mean_var_with_update,
          lambda: (ema.average(batch_mean), ema.average(batch_var))
      )
      normed = tf.nn.batch_normalization(
          x=input_var,
          mean=mean,
          variance=var,
          offset=beta,
          scale=gamma,
          variance_epsilon=epsilon,
          name="tf_bn"
      )
    return normed


def batch_norm_new(name, input_var, is_train, decay=0.999, epsilon=1e-5):
    """Batch normalization modified from BatchNormLayer in Tensorlayer.
    Source: <https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py#L2190>
    """

    inputs_shape = input_var.get_shape()
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]

    with tf.compat.v1.variable_scope(name) as scope:
        # Trainable beta and gamma variables
        beta = tf.compat.v1.get_variable('beta',
                                shape=params_shape,
                                initializer=tf.zeros_initializer())
        gamma = tf.compat.v1.get_variable('gamma',
                                shape=params_shape,
                                initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002))
        
        # Moving mean and variance updated during training
        moving_mean = tf.compat.v1.get_variable('moving_mean',
                                      params_shape,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_variance = tf.compat.v1.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False)
        
        # Compute mean and variance along axis
        batch_mean, batch_variance = tf.nn.moments(input_var, axis, name='moments')

        # Define ops to update moving_mean and moving_variance
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=False)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=False)

        # Define a function that :
        # 1. Update moving_mean & moving_variance with batch_mean & batch_variance
        # 2. Then return the batch_mean & batch_variance
        def mean_var_with_update():
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        # Perform different ops for training and testing
        if is_train:
            mean, variance = mean_var_with_update()
            normed = tf.nn.batch_normalization(input_var, mean, variance, beta, gamma, epsilon)
        
        else:
            normed = tf.nn.batch_normalization(input_var, moving_mean, moving_variance, beta, gamma, epsilon)
        # mean, variance = tf.cond(
        #     is_train,
        #     mean_var_with_update, # Training
        #     lambda: (moving_mean, moving_variance) # Testing - it will use the moving_mean and moving_variance (fixed during test) that are computed during training
        # )
        # normed = tf.nn.batch_normalization(input_var, mean, variance, beta, gamma, epsilon)

        return normed


def flatten(name, input_var):
    dim = 1
    for d in input_var.get_shape()[1:].as_list():
        dim *= d
    output_var = tf.reshape(input_var,
                            shape=[-1, dim],
                            name=name)

    return output_var
