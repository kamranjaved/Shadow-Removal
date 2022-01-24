import logging
import math

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from PIL import Image, ImageDraw

from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.gan_ops import *
from neuralgym.ops.summary_ops import *

def linear(input_, output_size, scope="linear", use_bias=True, bias_start=0.0, reuse=False, weight_norm=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        # weight normalization (Xiang & Li, 2017)
        if weight_norm == True:
            matrix = tf.nn.l2_normalize(matrix, [0])

        if use_bias == True:
            bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
            return tf.matmul(input_, matrix, name=scope) + bias
        else:
            return tf.matmul(input_, matrix, name=scope)


def conv2d(input_, output_channels, dilation=False, dilation_rate=1, ksize=3, stride=1, padding='SAME', scope="conv2d", use_bias=True, reuse=False, weight_norm=False):
    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('weights', [ksize, ksize, input_.get_shape()[-1], output_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        if weight_norm == True:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if dilation == True:
            conv = tf.nn.atrous_conv2d(input_, w, rate = dilation_rate, padding=padding)#, name=scope)#tf.layers.conv2d(input_, output_channels, ksize, strides=stride, padding=padding, dilation_rate= dilation_rate, name=scope)#conv = tf.nn.conv2d(input_, w, padding=padding, strides=[1, stride, stride, 1], dilations=[1, 1, 1, 1])#, name=scope)#
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding, name=scope)
        
        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
#--------------added by me for gated convolution-------------------

        x, y = tf.split(conv,2,3)
        x = tf.nn.elu(x)
        y = tf.nn.sigmoid(y)
        conv = x * y
            
        return conv

#-------------conv2d (not gated) for discriminator-----------------------------------
def conv2dd(input_, output_channels, dilation=False, dilation_rate=1, ksize=3, stride=1, padding='SAME', scope="conv2d", use_bias=True, reuse=False, weight_norm=False):
    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('weights', [ksize, ksize, input_.get_shape()[-1], output_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        if weight_norm == True:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if dilation == True:
            conv = tf.nn.atrous_conv2d(input_, w, rate = dilation_rate, padding=padding)#, name=scope)#tf.layers.conv2d(input_, output_channels, ksize, strides=stride, padding=padding, dilation_rate= dilation_rate, name=scope)#conv = tf.nn.conv2d(input_, w, padding=padding, strides=[1, stride, stride, 1], dilations=[1, 1, 1, 1])#, name=scope)#
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding, name=scope)
        
        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
#--------------added by me for gated convolution-------------------
        '''
        x, y = tf.split(conv,2,3)
        x = tf.nn.elu(x)
        y = tf.nn.sigmoid(y)
        conv = x * y
        '''
            
        return conv





def dilated_conv2d(input_, output_channels, dilation, ksize=3, stride=1, padding='SAME', scope="conv2d", use_bias=True, reuse=False):#, weight_norm=False):
    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('weights', [ksize, ksize, input_.get_shape()[-1], output_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        #if weight_norm == True:
        #    w = tf.nn.l2_normalize(w, [0, 1, 2])

        conv = tf.nn.atrous_conv2d(input_, w, rate=dilation, padding=padding, name=scope)
        
        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
#--------------added by me for gated convolution-------------------
        '''
        x, y = tf.split(conv,2,3)
        x = tf.nn.elu(x)
        y = tf.nn.sigmoid(y)
        conv = x * y
        '''
            
        return conv


def deconv2d(input_, output_channels, ksize=3, stride=2, padding='SAME', scope="deconv2d", use_bias=True, reuse=False):#, weight_norm=False):
    shape = input_.get_shape().as_list()
    output_shape = tf.stack([tf.shape(input_)[0], shape[1]*stride, shape[2]*stride, output_channels])

    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [ksize, ksize, output_channels, input_.get_shape()[-1]], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        #if weight_norm == True:
        #    w = tf.nn.l2_normalize(w, [0, 1, 2])

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding, name=scope)

        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)
        else:
            deconv = tf.reshape(deconv, output_shape)
        
        return deconv


def dilated_deconv2d(input_, output_channels, dilation, ksize=3, stride=2, padding='SAME', scope="deconv2d", use_bias=True, reuse=False):#, weight_norm=False):
    shape = input_.get_shape().as_list()
    output_shape = tf.stack([tf.shape(input_)[0], shape[1]*stride, shape[2]*stride, output_channels])

    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [ksize, ksize, output_channels, input_.get_shape()[-1]], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        #if weight_norm == True:
        #    w = tf.nn.l2_normalize(w, [0, 1, 2])

        deconv = tf.nn.atrous_conv2d_transpose(input_, w, output_shape=output_shape, rate=dilation, padding=padding, name=scope)

        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)
        else:
            deconv = tf.reshape(deconv, output_shape)
        
        return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name+'_lrelu')


def prelu(x, init=0.2, reuse=False, name='prelu'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        t = tf.get_variable("tangent", [1], tf.float32, initializer=tf.constant_initializer(init))
    return tf.add(tf.nn.relu(x), tf.multiply(t,tf.minimum(x,0)),name=name+'_prelu')


def batch_normalization(x, train_mode, epsilon=1e-6, decay = 0.9, name="batch_norm", use_vars=True, reuse=False):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if use_vars:
            beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
            gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.constant_initializer(1.))
        else:
            beta = None
            gamma = None
        mean = tf.get_variable("mean", [shape[-1]], initializer=tf.constant_initializer(0.), trainable=False)
        var = tf.get_variable("var", [shape[-1]], initializer=tf.constant_initializer(1.), trainable=False)
        try:
            batch_mean, batch_var = tf.nn.moments(x,[0, 1, 2])
        except:
            batch_mean, batch_var = tf.nn.moments(x,[0])

    train_mean = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(var, var * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
        train_bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon, name=name)
    inference_bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name=name)

    return tf.cond(train_mode, lambda: train_bn, lambda: inference_bn)


def affine(x, reuse=False, name='affine'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.constant_initializer(1.))
    return tf.add(tf.multiply(x, gamma),beta, name=name+'_affine')


def tprelu(x, init=0.2, reuse=False, name='tprelu'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        t = tf.get_variable("tangent", [1], tf.float32, initializer=tf.constant_initializer(init))
        a = tf.get_variable("translation", shape[-1], tf.float32, initializer=tf.constant_initializer(0.))
    x = x - a
    p = tf.add(tf.nn.relu(x), tf.multiply(t,tf.minimum(x,0)))
    return tf.add(p,a,name=name+'_tprelu')


def tlrelu(x, leak=0.2, name="tlrelu"):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        a = tf.get_variable("translation", shape[-1], tf.float32, initializer=tf.constant_initializer(0.))
    x = x - a
    return tf.add(tf.maximum(x, leak*x), a, name=name+'_tlrelu')


def avg_pool(x, ksize=2, strides=2, padding='SAME', name="avg_pool"):
    return tf.nn.avg_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding=padding, name=name)


def instance_normalization(x, train_mode=True, epsilon=1e-6, decay = 0.9, name="instance_norm", use_vars=True, reuse=False):
    shape = x.get_shape().as_list()
    if len(shape) == 2:
        return x
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if use_vars:
            beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
            gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.constant_initializer(1.))
        else:
            beta = None
            gamma = None
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)

    normalized = tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(variance, epsilon)))
    if use_vars:
        normalized = tf.add(tf.multiply(gamma, normalized), beta)

    return normalized


def residual_block(x, f_size=128, ksize=3, norm='instance', name="residual", train_mode=True, reuse=False):
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    h = conv2d(x, f_size, ksize=ksize, stride=1, padding='SAME', scope=name+'_conv0', use_bias='False', reuse=reuse)
    h = batch_norm(h, name=name+'_bn0', train_mode=train_mode, reuse=reuse)
    h = tf.nn.relu(h, name=name+'_conv0')
    h = conv2d(h, f_size, ksize=ksize, stride=1, padding='SAME', scope=name+'_conv1', use_bias='False', reuse=reuse)
    h = batch_norm(h, name=name+'_bn1', train_mode=train_mode, reuse=reuse)
    return x + h


def dense_block(x, growth_rate=4, ksize=3, n_layers=4, bottleneck=False, norm='instance', rect='lrelu', name="dense", train_mode=True, reuse=False):
    layers = []
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    if rect == 'relu':
        rectifier = tf.nn.relu
    else:
        rectifier = lrelu

    layers.append(x)
    for i in range(n_layers):
        with tf.variable_scope(name+str(i)):
            normalized = batch_norm(layers[-1], train_mode=train_mode, reuse=reuse)
            rectified = rectifier(normalized)
            if bottleneck:
                rectified = conv2d(rectified, 4*growth_rate, ksize=1, scope='bottelneck', reuse=reuse)
            convolved = conv2d(rectified, growth_rate, ksize=ksize, reuse=reuse)
            concated = tf.concat([layers[-1], convolved], axis=3)
            layers.append(concated)

    return layers[-1]
def se_block(x, reduction_ratio=16, scope='SE', rect='relu', reuse=False):
    shape = x.get_shape().as_list()
    channels = shape[3]
    reducted = int(channels/reduction_ratio)

    if rect == 'lrelu':
        rectifier = lrelu
    else:
        rectifier = tf.nn.relu
    
    with tf.variable_scope(scope) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # Sqeeze (using average)
        h = tf.nn.avg_pool(x, ksize=[1, shape[1], shape[2], 1], strides=[1,1,1,1], padding='VALID')
        h = tf.reshape(h, [-1, channels])
        # Excitation 1
        h = linear(h, reducted, scope="excitation1", reuse=reuse)
        h = rectifier(h)
        # Excitation 2
        h = linear(h, channels, scope="excitation2", reuse=reuse)
        h = tf.nn.sigmoid(h)
        # Scale
        h = tf.reshape(h, [-1, 1, 1, channels])
        y = tf.multiply(h, x, name='scale')

    return y


#def erosion(input_, output_channels, dilation=False, dilation_rate=1, ksize=3, stride=1, padding='SAME', scope="conv2d", use_bias=True, reuse=False, weight_norm=False):
def erosion(input_, ksize=3, stride=1, rate = 3, padding='SAME', name='erosion'):
    with tf.variable_scope(name):
        #w = [ksize, ksize, input_.get_shape()[-1]]
        erod = tf.nn.erosion2d(input_, kernel = [ksize, ksize, 3], strides=[1, stride, stride, 1], rates = [1, rate, rate, 1], padding=padding, name=name)
            
        return erod


#---------from gated convolution--------to use gated convolution_only___


logger = logging.getLogger()
np.random.seed(2018)


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):  #gated convolution
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
    if cnum == 3 or activation is None:
        # conv for output
        return x
    x, y = tf.split(x, 2, 3)
    x = activation(x)
    y = tf.nn.sigmoid(y)
    x = x * y
    return x


@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    x = conv2d_spectral_norm(x, cnum, ksize, stride, 'SAME', name=name)
    x = tf.nn.leaky_relu(x)
    return x
#--------------------------------------------------------------------------

