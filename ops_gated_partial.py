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

import tensorflow.contrib as tf_contrib

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

#-----------gated convolution---------------------------------------
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

#-------------conv2d (not gated) -----------------------------------
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



##################################################################################
# Partial Layer
##################################################################################



weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)


def partial_conv(x, mask, channels, kernel=3, stride=2, use_bias=True, padding='SAME', scope='conv_0'):
    with tf.variable_scope(scope):
        if padding.lower() == 'SAME'.lower() :
            with tf.variable_scope('mask'):
                _, h, w, _ = x.get_shape().as_list()
                print('Shape of mask before inside partial convolution', np.shape(mask))
                print('Shape of partial conv before inside partial convolution', np.shape(x))

                slide_window = kernel * kernel
                #mask = tf.ones(shape=[1, h, w, 1])
                #print('Shape of mask inside partial convolution', np.shape(mask))

                update_mask = tf.layers.conv2d(mask, filters=1,
                                               kernel_size=kernel, kernel_initializer=tf.constant_initializer(1.0),
                                               strides=stride, padding=padding, use_bias=False, trainable=False)

                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                print('Shape of update_mask partial convolution', np.shape(update_mask))
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('x'):
                x = tf.layers.conv2d(x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, padding=padding, use_bias=False)

                print('Shape of partial conv inside partial convolution', np.shape(x))
                print('Shape of mask_ratio inside partial convolution', np.shape(mask_ratio))
                x = x * mask_ratio

                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
                    x = x * update_mask

        else :
            x = tf.layers.conv2d(x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, padding=padding, use_bias=use_bias)

        #return x
        return x, update_mask




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


#--------------------------contextual attention----------------------------


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize

'''
def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_bilinear)
    return y, flow

'''
#------------------------------------------------------------------------------------------------------------------------


