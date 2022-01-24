import numpy as np
import tensorflow as tf
from ops_gated_partial import *


#---------------------------------------Unet encoder with simple convolution---------------------------------------------

#def unet(generator_inputs, generator_outputs_channels=3, n_layers=3, ngf=64, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, n_dropout=3, use_se_block=True, use_full_se=True):
def unet_encoder(generator_inputs, generator_outputs_channels=3, n_layers=3, ngf=64, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, n_dropout=3, use_se_block=True, use_full_se=True):
    # n_layers: # of layers with ngf*8
    # 256: 5 layers
    # 128: 4 layers
    if use_se_block != True:
        use_full_se = False
    layers = []

    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope(name+"encoder_1"):
        output = conv2dd(generator_inputs, ngf, ksize=ksize, stride=1, reuse=reuse)#2)
        layers.append(output)
        print('Conv layer with stride=1 encoder_%d' % (len(layers) ) )

    layer_specs = [ngf*2, ngf*4]
    for i in range(n_layers):
        layer_specs.append(ngf*8)

    for out_channels in layer_specs:
        with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv2dd(rectified, out_channels, ksize=ksize, stride=2, reuse=reuse)
            output = batch_norm(convolved, train_mode=train_mode, reuse=reuse)
            layers.append(output)
            print('Conv layer with stride=2 encoder_%d' % (len(layers)) )
            print('Shape of output', np.shape(convolved))
    layer_specs = []
    for i in range(n_layers-1):
        if i < n_dropout:
            layer_specs.append((ngf*8, 0.5))
        else:
            layer_specs.append((ngf*8, 0.0))
    layer_specs = layer_specs + [(ngf*4, 0.0), (ngf*2, 0.0), (ngf, 0.0)]

    num_encoder_layers = len(layers)

    return layers[0],layers[1],layers[2],layers[3], layers[-1], num_encoder_layers  #layers[-1]=last layer output


def unet_bottleneck(generator_inputs, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, use_se_block=True, use_full_se=True):


    if use_se_block != True:
        use_full_se = False
    layers = []
    layers_inp = [generator_inputs]
    print("bottle neck input", np.shape(layers_inp[-1]))

    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    with tf.variable_scope(name+"BN_conv1"):
         rectified = lrelu(layers_inp[-1], 0.2) #AC = conv2dd(layers_inp[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=2)
         convolved = conv2dd(rectified, 512, ksize=ksize, stride=1, reuse=reuse)
         output = batch_norm(convolved, train_mode=train_mode, reuse=reuse)
         layers.append(output)
         print('bottleneck Conv1 layer with stride=2 encoder_%d' % (len(layers)) )
         #print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))

    with tf.variable_scope(name+"BN_conv2"):
         rectified = lrelu(layers_inp[-1], 0.2) #AC = conv2dd(layers_inp[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=2)
         convolved = conv2dd(rectified, 256, ksize=ksize, stride=1, reuse=reuse)
         output = batch_norm(convolved, train_mode=train_mode, reuse=reuse)
         layers.append(output)
         print('bottleneck Conv2 layer with stride=2 encoder_%d' % (len(layers)) )
         #print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))

    with tf.variable_scope(name+"BN_conv3"):
         rectified = lrelu(layers_inp[-1], 0.2) #AC = conv2dd(layers_inp[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=2)
         convolved = conv2dd(rectified, 128, ksize=ksize, stride=1, reuse=reuse)
         output = batch_norm(convolved, train_mode=train_mode, reuse=reuse)
         layers.append(output)
         print('bottleneck Conv3 layer with stride=2 encoder_%d' % (len(layers)) )
         #print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))

    with tf.variable_scope(name+"BN_1"):
         AC = conv2dd(layers_inp[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=2)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         #print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))

    with tf.variable_scope(name+"BN_2"):
         AC = conv2dd(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=4)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    with tf.variable_scope(name+"BN_3"):
         num_encoder_layers = len(layers)
         AC = conv2dd(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=8)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    with tf.variable_scope(name+"BN_4"):
         AC = conv2dd(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=16)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))

    num_encoder_layers = len(layers)
    return layers[0],layers[1],layers[2],layers[3],layers[4],layers[5],layers[6],layers[-1], num_encoder_layers  #layers[-1]=last layer output

#---------------------------------------Unet encoder with gated convolution---------------------------------------------

#def unet(generator_inputs, generator_outputs_channels=3, n_layers=3, ngf=64, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, n_dropout=3, use_se_block=True, use_full_se=True):
def unet_encoder_gated(generator_inputs, generator_outputs_channels=3, n_layers=3, ngf=64, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, n_dropout=3, use_se_block=True, use_full_se=True):
    # n_layers: # of layers with ngf*8
    # 256: 5 layers
    # 128: 4 layers
    if use_se_block != True:
        use_full_se = False
    layers = []
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope(name+"encoder_1"):
        output = conv2d(generator_inputs, ngf, ksize=ksize, stride=1, reuse=reuse)#2)
        layers.append(output)
        print('Conv layer with stride=1 encoder_%d' % (len(layers) ) )

    layer_specs = [ngf*2, ngf*4]
    for i in range(n_layers):
        layer_specs.append(ngf*8)

    for out_channels in layer_specs:
        with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv2d(rectified, out_channels, ksize=ksize, stride=2, reuse=reuse)
            output = batch_norm(convolved, train_mode=train_mode, reuse=reuse)
            layers.append(output)
            print('Conv layer with stride=2 encoder_%d' % (len(layers)) )
            print('Shape of output', np.shape(convolved))
    layer_specs = []
    for i in range(n_layers-1):
        if i < n_dropout:
            layer_specs.append((ngf*8, 0.5))
        else:
            layer_specs.append((ngf*8, 0.0))
    layer_specs = layer_specs + [(ngf*4, 0.0), (ngf*2, 0.0), (ngf, 0.0)]

    '''
    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         AC = conv2d(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=2)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))

    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         AC = conv2d(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=4)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         num_encoder_layers = len(layers)
         AC = conv2d(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=8)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         AC = conv2d(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=16)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    '''

    num_encoder_layers = len(layers)
    return layers[0],layers[1],layers[2],layers[3],layers[-1], num_encoder_layers  #layers[-1]=last layer output


def unet_encoder_partial(generator_inputs, mask, generator_outputs_channels=3, n_layers=3, ngf=64, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, n_dropout=3, use_se_block=True, use_full_se=True):
    # n_layers: # of layers with ngf*8
    # 256: 5 layers
    # 128: 4 layers
    if use_se_block != True:
        use_full_se = False
    layers = []
    layers_mask = []
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope(name+"encoder_1"):
        output, mask = partial_conv(generator_inputs, mask, ngf, kernel=ksize, stride=1)#2)
        layers.append(output)
        layers_mask.append(mask)
        print('Partial Conv layer with stride=1 encoder_%d' % (len(layers) ) )

    layer_specs = [ngf*2, ngf*4]
    for i in range(n_layers):
        layer_specs.append(ngf*8)

    for out_channels in layer_specs:
        with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved, mask = partial_conv(rectified, layers_mask[-1], out_channels, kernel=ksize, stride=2)
            output = batch_norm(convolved, train_mode=train_mode, reuse=reuse)
            layers.append(output)
            layers_mask.append(mask)
            print('Partial Conv layer with stride=2 encoder_%d' % (len(layers)) )
            print('Shape of output in Partial conv', np.shape(convolved))
    layer_specs = []
    for i in range(n_layers-1):
        if i < n_dropout:
            layer_specs.append((ngf*8, 0.5))
        else:
            layer_specs.append((ngf*8, 0.0))
    layer_specs = layer_specs + [(ngf*4, 0.0), (ngf*2, 0.0), (ngf, 0.0)]

    '''    
    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         AC = conv2dd(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=2)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))

    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         AC = conv2dd(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=4)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         num_encoder_layers = len(layers)
         AC = conv2dd(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=8)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    with tf.variable_scope(name+"encoder_%d" % (len(layers) + 1)):
         AC = conv2dd(layers[-1], 128, ksize=ksize, reuse=reuse, dilation=True, dilation_rate=16)
         layers.append(AC)
         print('Atrous Conv layer with stride=2 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(AC))
         if use_full_se:
            output = se_block(layers[-1], reduction_ratio=16, rect='lrelu', reuse=reuse)#----SE---
         layers.append(output)
         print('SE layer with rr=16 encoder_%d' % (len(layers)) )
         print('Shape of output', np.shape(output))
    '''
  

    num_encoder_layers = len(layers)
    return layers[0],layers[1],layers[2],layers[3],layers[-1], num_encoder_layers  #layers[-1]=last



#---------------------decoder for gated one -----------------------------------

def unet_decoder_gated(layers_inp, layer0,layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8, layer9,layer10,layer11,layerlast, num_encoder_layers=5, generator_outputs_channels=3, n_layers=3, ngf=64, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, n_dropout=3, use_se_block=True, use_full_se=True):
    # n_layers: # of layers with ngf*8
    # 256: 5 layers
    # 128: 4 layers
    if use_se_block != True:
        use_full_se = False
    layers = [layer0,layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8, layer9,layer10,layer11,layerlast]
    print("shape of decoder -1 layer", np.shape(layers[-1]))
    print("shape shape 4 partial conv layer for decoder layer", np.shape(layers[4]))
    print("shape shape 3 partial conv layer for decoder layer", np.shape(layers[3]))
    print("shape shape 2 partial conv layer for decoder layer", np.shape(layers[2]))
    print("shape shape 1 partial conv layer for decoder layer", np.shape(layers[1]))
    print("shape shape 0 partial conv layer for decoder layer", np.shape(layers[0]))
    print("shape of decoder -1 layer input concatenated", np.shape(layers_inp))

    #print('layers variable', np.shape(layers[-1]))
    #print('encoder output[-1]', np.shape(layers[0]))
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    #layer_specs = [ngf*2, ngf*4]
    #for i in range(n_layers):
        #layer_specs.append(ngf*8)

    layers_inp = [layers_inp]

    layer_specs = []
    for i in range(n_layers-1):
        if i < n_dropout:
            layer_specs.append((ngf*8, 0.5))
        else:
            layer_specs.append((ngf*8, 0.0))
    layer_specs = layer_specs + [(ngf*4, 0.0), (ngf*2, 0.0), (ngf, 0.0)]

    
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 9
        print('skip_layer number', skip_layer)
        print('out channel number', out_channels)
        with tf.variable_scope(name+"decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers_inp[-1]
                print('Shape of decoder layer o input', np.shape(input))
            else:
                input = tf.concat([layers_inp[-1], layers[skip_layer]], axis=3)
                print('concatenated shape  ', np.shape(input))
            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv2d(rectified, out_channels, ksize, stride=2, reuse=reuse)
            output = batch_norm(output, train_mode=train_mode, reuse=reuse)


            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers_inp.append(output)
            print('DeConv layer with stride=2 decoder_%d' % (len(layers)) )
            print('Shape of output', np.shape(output))

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope(name+"decoder_1"):
        input = tf.concat([layers_inp[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv2d(rectified, generator_outputs_channels, ksize, stride=1, reuse=reuse)
        output = tf.tanh(output)
        layers_inp.append(output)
        print('DeConv layer with stride=1 d_1 decoder_%d' % (len(layers)) )
        print('Shape of output', np.shape(output))

    return layers_inp[-1]




#------------------------------------------------------------------------------


def unet_decoder(layer0,layer1,layer2,layer3,layer4,layer5,layer6,layer7, layer8, layer9,layer10,layer11,layerlast, num_encoder_layers=5, generator_outputs_channels=3, n_layers=3, ngf=64, ksize=4, norm='instance', name='unet_', train_mode=True, reuse=False, n_dropout=3, use_se_block=True, use_full_se=True):
    # n_layers: # of layers with ngf*8
    # 256: 5 layers
    # 128: 4 layers
    if use_se_block != True:
        use_full_se = False
    layers = [layer0,layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8, layer9,layer10,layer11,layerlast]
    print("shape of decoder -1 layer", np.shape(layers[-1]))
    #print('layers variable', np.shape(layers[-1]))
    #print('encoder output[-1]', np.shape(layers[0]))
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization
    '''
    layers[-1] = layer
    layers[0] = layer0
    layers[1] = layer1
    layers[2] = layer2
    layers[3] = layer3
    layers[4] = layer4

    layers[5] = layer5
    layers[6] = layer6
    layers[7] = layer7
    '''
    #layer_specs = [ngf*2, ngf*4]
    #for i in range(n_layers):
        #layer_specs.append(ngf*8)

    layer_specs = []
    for i in range(n_layers-1):
        if i < n_dropout:
            layer_specs.append((ngf*8, 0.5))
        else:
            layer_specs.append((ngf*8, 0.0))
    layer_specs = layer_specs + [(ngf*4, 0.0), (ngf*2, 0.0), (ngf, 0.0)]

    
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 9
        print('skip_layer number', skip_layer)
        with tf.variable_scope(name+"decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                print('concatenated shape  ', np.shape(input))
            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv2d(rectified, out_channels, ksize, stride=2, reuse=reuse)
            output = batch_norm(output, train_mode=train_mode, reuse=reuse)


            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)
            print('DeConv layer with stride=2 decoder_%d' % (len(layers)) )
            print('Shape of output', np.shape(output))

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope(name+"decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv2d(rectified, generator_outputs_channels, ksize, stride=1, reuse=reuse)
        output = tf.tanh(output)
        layers.append(output)
        print('DeConv layer with stride=1 d_1 decoder_%d' % (len(layers)) )
        print('Shape of output', np.shape(output))

    return layers[-1]

def resnet(generator_inputs, generator_outputs_channels=3, n_blocks=6, ngf=32, ksize=3, norm='instance', name='resnet_', train_mode=True, reuse=False, n_dropout=0):
    # resnet do not use ksize
    f_num = ngf*4  # for resnet_blocks
    layers = []
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    # conv0 -c7s1-32
    convolved = conv2d(generator_inputs, ngf, ksize=7, stride=1, padding='SAME', scope=name+'conv0', use_bias='False', reuse=reuse)
    normalized = batch_norm(convolved, name=name+'conv0', use_vars=True, train_mode=train_mode, reuse=reuse)
    rectified = tf.nn.relu(normalized, name=name+'conv0')
    layers.append(rectified)

    # downsampling -d64, d128
    for i in [1,2]:
        convolved = conv2d(layers[-1], ngf*2*i, ksize=3, stride=2, padding='SAME', scope=name+'d'+str(i), use_bias='False', reuse=reuse)
        normalized = batch_norm(convolved, name=name+'d'+str(i), use_vars=True, train_mode=train_mode, reuse=reuse)
        rectified = tf.nn.relu(normalized, name=name+'d'+str(i))
        layers.append(rectified)

    # residual blocks -R128
    for i in range(n_blocks):
        res_block = residual_block(layers[-1], f_size=f_num, ksize=3, norm=norm, name=name+'residual_block'+str(i), reuse=reuse)
        layers.append(res_block)

    # upsampling - u64, u32
    for i in [1,2]:
        convolved = deconv2d(layers[-1], int(ngf*2/i), ksize=3, stride=2, padding='SAME', scope=name+'u'+str(i), use_bias='False', reuse=reuse)
        normalized = batch_norm(convolved, name=name+'u'+str(i), use_vars=True, train_mode=train_mode, reuse=reuse)
        rectified = tf.nn.relu(normalized, name=name+'u'+str(i))
        layers.append(rectified)

    # conv1 -c7s1-32
    convolved = conv2d(layers[-1], generator_outputs_channels, ksize=7, stride=1, padding='SAME', scope=name+'conv1', use_bias='False', reuse=reuse)
    normalized = batch_norm(convolved, name=name+'conv1', use_vars=True, train_mode=train_mode, reuse=reuse)
    rectified = tf.tanh(normalized, name=name+'conv1')
    layers.append(rectified)

    return layers[-1]

def densenet(generator_inputs, generator_outputs_channels=3, n_blocks=4, ngf=16, growth_rate=16, dense_block_size=4, bottleneck=True, norm='instance', name='densenet_', train_mode=True, reuse=False):
    f_num = ngf*4  # for resnet_blocks
    layers = []
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    # conv0 -c7s1-32
    convolved = conv2d(generator_inputs, ngf, ksize=7, stride=1, padding='SAME', scope=name+'conv0', use_bias='False', reuse=reuse)
    normalized = batch_norm(convolved, name=name+'conv0', use_vars=True, train_mode=train_mode, reuse=reuse)
    rectified = tf.nn.relu(normalized, name=name+'conv0')
    layers.append(rectified)

    # downsampling -d64, d128
    for i in [1,2]:
        convolved = conv2d(layers[-1], ngf*2*i, ksize=3, stride=2, padding='SAME', scope=name+'d'+str(i), use_bias='False', reuse=reuse)
        normalized = batch_norm(convolved, name=name+'d'+str(i), use_vars=True, train_mode=train_mode, reuse=reuse)
        rectified = tf.nn.relu(normalized, name=name+'d'+str(i))
        layers.append(rectified)

    # dense blocks -R128
    for i in range(n_blocks):
        dense = dense_block(layers[-1], growth_rate, 3, dense_block_size, bottleneck=bottleneck, norm=norm, name=name+'dense_block'+str(i), reuse=reuse)
        transition = conv2d(dense, f_num, ksize=1, scope=name+'transition'+str(i), use_bias='True', reuse=reuse)
        layers.append(transition)

    # upsampling - u64, u32
    for i in [1,2]:
        convolved = deconv2d(layers[-1], int(ngf*2/i), ksize=3, stride=2, padding='SAME', scope=name+'u'+str(i), use_bias='False', reuse=reuse)
        normalized = batch_norm(convolved, name=name+'u'+str(i), use_vars=True, train_mode=train_mode, reuse=reuse)
        rectified = tf.nn.relu(normalized, name=name+'u'+str(i))
        layers.append(rectified)

    # conv1 -c7s1-32
    convolved = conv2d(layers[-1], generator_outputs_channels, ksize=7, stride=1, padding='SAME', scope=name+'conv1', use_bias='False', reuse=reuse)
    normalized = batch_norm(convolved, name=name+'conv1', use_vars=True, train_mode=train_mode, reuse=reuse)
    rectified = tf.tanh(normalized, name=name+'conv1')
    layers.append(rectified)

    return layers[-1]

def pixel(inputs, n_layers = 3, ndf=64, ksize=4, norm='instance', name='pixel70_', train_mode=True, reuse=False, sigmoid_output=True, use_se_block=True, use_full_se=True):
    layers = []
    if norm == 'instance':
        batch_norm = instance_normalization
    else:
        batch_norm = batch_normalization

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    #inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope(name+"layer_1"):
        convolved = conv2dd(inputs, ndf, ksize=ksize, stride=2, reuse=reuse)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)
        print('Conv layer with stride=2 Ddiscriminator_%d' % (len(layers)) )
        print('Shape of output', np.shape(convolved))
    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope(name+"layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv2dd(layers[-1], out_channels, ksize=ksize, stride=stride, reuse=reuse)
            normalized = batch_norm(convolved, train_mode=train_mode, reuse=reuse)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)
            print('Conv layer with stride=2 discriminator_%d' % (len(layers)) )
            print('Shape of output', np.shape(convolved))
    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope(name+"layer_%d" % (len(layers) + 1)):
        convolved = conv2dd(rectified, output_channels=1, ksize=ksize, stride=1, reuse=reuse)
        if sigmoid_output:
            output = tf.sigmoid(convolved)
        else:
            output = convolved
        layers.append(output)
        print('Conv layer with stride=2 Discriminator_%d' % (len(layers)) )
        print('Shape of output', np.shape(convolved))
    return layers[-1]

def GAN_loss(predict_fake, predict_real, mode='log', eps=1e-6):
    if mode == 'ls':
        # least square adversarial loss
        d_loss = tf.add(tf.reduce_mean((predict_real-1)**2), tf.reduce_mean((predict_fake)**2), name='d_loss')
        g_loss = tf.reduce_mean((predict_fake-1)**2, name='g_loss')
        print("Least Square GAN")
    elif mode == 'hinge':
        d_loss = tf.add(-tf.reduce_mean(tf.minimum(0,predict_real-1) + tf.minumum(0,-1-predict_real)), -tf.reduce_mean(tf.minimum(0,predict_fake-1) + tf.minumum(0,-1-predict_fake)), name='d_loss')
        g_loss = -tf.reduce_mean(predict_fake+ tf.minumum(0,-1-predict_fake), name='g_loss')
    else:
        # log adversarial loss
        d_loss = tf.add(-tf.reduce_mean(tf.log(predict_real + eps)), -tf.reduce_mean(tf.log(1 - predict_fake + eps)), name='d_loss')
        g_loss = tf.reduce_mean(-tf.log(predict_fake + eps), name='g_loss')

    Gen_loss_sum = tf.summary.scalar("Generator_adversarial_loss", g_loss)
    Dis_loss_sum = tf.summary.scalar("Discriminator_adversarial_loss", d_loss)

    return g_loss, d_loss

def ReviewGAN_loss(cur_fake, cur_real, rgen_fake, rdis_real, rdis_fake, use_review, mode='log', eps=1e-6):
    cur_batch_size = cur_fake.get_shape().as_list()[0]
    re_batch_size = rgen_fake.get_shape().as_list()[0]
    cur_w = float(cur_batch_size)/float(cur_batch_size+re_batch_size)
    re_w = float(re_batch_size)/float(cur_batch_size+re_batch_size)

    cur_g_loss, cur_d_loss = GAN_loss(cur_fake, cur_real, mode=mode)
    
    # review losses
    if mode == 'ls':
        # least square adversarial loss
        re_d_loss = tf.add(tf.reduce_mean((rdis_real-1)**2), tf.reduce_mean((rdis_fake)**2), name='d_loss')
        re_g_loss = tf.reduce_mean((rgen_fake-1)**2, name='g_loss')
        print("Least Square GAN")
    else:
        # log adversarial loss
        re_d_loss = tf.add(-tf.reduce_mean(tf.log(rdis_real + eps)), -tf.reduce_mean(tf.log(1 - rdis_fake + eps)), name='d_loss')
        re_g_loss = tf.reduce_mean(-tf.log(rgen_fake + eps), name='g_loss')

    g_loss = tf.cond(use_review, lambda: ((cur_w*cur_g_loss)+(re_w*re_g_loss)), lambda: cur_g_loss)
    d_loss = tf.cond(use_review, lambda: ((cur_w*cur_d_loss)+(re_w*re_d_loss)), lambda: cur_d_loss)

    Gen_loss_sum = tf.summary.scalar("Generator_adversarial_loss", g_loss)
    Dis_loss_sum = tf.summary.scalar("Discriminator_adversarial_loss", d_loss)

    return g_loss, d_loss


