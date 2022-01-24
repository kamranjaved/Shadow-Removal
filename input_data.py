import os
import numpy as np
import tensorflow as tf
import random

def load(inputs_dir, queue_capacity, img_size, channel=3, name='', mode='train', paired=True):
    with tf.device('/cpu:0'):
        # Create a list of filenames
        filenames = os.listdir(inputs_dir)
        filenames.sort()
        # Check jpeg
        img_filenames = []
        img_filedirs = []
        for i in range(len(filenames)):
            if (filenames[i][-4:] == '.jpg') or (filenames[i][-4:] == '.JPG') or (filenames[i][-5:] == '.jpeg') or (filenames[i][-4:] == '.png') or (filenames[i][-4:] == '.PNG'):
                img_filedirs.append(inputs_dir + '/' + filenames[i])
                img_filenames.append(filenames[i])
        num_data = len(img_filedirs)
        print('number of data: %d' % num_data)

        # Create a queue that produces the filenames to read
        filedir_queue = tf.train.string_input_producer(img_filedirs, shuffle=False, capacity=queue_capacity, name=name+'_filename_queue')
        if mode == 'test':
            filename_queue = tf.train.string_input_producer(img_filenames, shuffle=False, capacity=queue_capacity, name=name+'_filename_queue')
            filename_op = filename_queue.dequeue()
        else:
            filename_op = None
        # Create a reader for the filequeue
        reader = tf.WholeFileReader()
        # Read in the files
        _, image_file = reader.read(filedir_queue)
        # Convert the Tensor(of type string) to representing the Tensor of type uint8
        # and shape [height, width, channels] representing the images
        image = tf.image.decode_image(image_file, channels=channel)
        # set shape of image (for next process)
        if paired:
            image.set_shape([img_size, 4*img_size, channel])
        else:
            image.set_shape([img_size, img_size, channel])
    
    return image, num_data, filename_op

def preprocess(image, img_size=256, channel=3, flip=False, crop_size=None, random_crop=False, paired=True, seed=None):
    with tf.device('/cpu:0'):
        if paired:
            # normalize
            image = tf.subtract(tf.to_float(image), 127.5)
            image = tf.divide(image, 127.5)          
            # divide image A and B
            imgA, imgB, imgC, imgD = tf.split(image, 4, axis=1)
            # crop
            if seed == None:
                seed = random.randint(0, 2**31 - 1)
            if (crop_size != None) and (crop_size != False) and (crop_size != img_size):
                if random_crop:
                    imgA = tf.random_crop(imgA, [crop_size, crop_size, channel], seed=seed)
                    imgB = tf.random_crop(imgB, [crop_size, crop_size, channel], seed=seed)
                    imgC = tf.random_crop(imgC, [crop_size, crop_size, channel], seed=seed)
                    imgD = tf.random_crop(imgD, [crop_size, crop_size, channel], seed=seed)
                else:
                    imgA = tf.image.resize_image_with_crop_or_pad(imgA, crop_size, crop_size)
                    imgB = tf.image.resize_image_with_crop_or_pad(imgB, crop_size, crop_size)
                    imgC = tf.image.resize_image_with_crop_or_pad(imgC, crop_size, crop_size)
                    imgD = tf.image.resize_image_with_crop_or_pad(imgD, crop_size, crop_size)
                imgA.set_shape([crop_size, crop_size, channel])
                imgB.set_shape([crop_size, crop_size, channel])
                imgC.set_shape([crop_size, crop_size, channel])
                imgD.set_shape([crop_size, crop_size, channel])
            # flip
            if flip == True:
                imgA = tf.image.random_flip_left_right(imgA, seed=seed)
                imgB = tf.image.random_flip_left_right(imgB, seed=seed)
                imgC = tf.image.random_flip_left_right(imgC, seed=seed)
                imgD = tf.image.random_flip_left_right(imgD, seed=seed)
            return imgA, imgB, imgC, imgD
        else:
            # normalize
            image = tf.subtract(tf.to_float(image), 127.5)
            image = tf.divide(image, 127.5)
            # crop
            if seed == None:
                seed = random.randint(0, 2**31 - 1)
            if (crop_size != None) and (crop_size != False) and (crop_size != img_size):
                if random_crop:
                    image = tf.random_crop(image, [crop_size, crop_size, channel], seed=seed)
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, img_size)
                image.set_shape([crop_size, crop_size, channel])
            # flip
            if flip == True:
                image = tf.image.random_flip_left_right(image, seed=seed)
            # other argumentations
            #image = tf.image.random_brightness(image, max_delta=0.2)
            #image = tf.image.random_contrast(image, lower=0.2, upper=1.2)
            #tf.image.random_hue(image, max_delta=0.5)
            return image   


def batch_inputs(input_dir, batch_size, img_size=256, name='', channelA = 3, channelB = 3, channelC = 3, channelD = 3, mode='train', flip=False, crop_size=None, random_crop=False):
    if mode == 'test':
        flip = False
    if (channelA == 1) and (channelB == 1):
        channel = 1
    else:
        channel = 3
    with tf.device('/cpu:0'):
        # load and preprocess images
        image, num_data, filename = load(input_dir, 16*batch_size, img_size, channel, name=name, mode=mode)
        # preprocessing
        imgA, imgB, imgC, imgD = preprocess(image, img_size, channel, flip, crop_size, random_crop)
        if channel == 3:
            if channelA == 1:
                imgA = tf.image.rgb_to_grayscale(imgA)
            if channelB == 1:
                imgB = tf.image.rgb_to_grayscale(imgB)
            if channelC == 1:
                imgC = tf.image.rgb_to_grayscale(imgC)
            if channelD == 1:
                imgD = tf.image.rgb_to_grayscale(imgD)
        # make batch
        if mode == 'train':
            batchA, batchB, batchC, batchD = tf.train.shuffle_batch([imgA, imgB, imgC, imgD], batch_size=batch_size, num_threads = 4, capacity=16*batch_size, min_after_dequeue=8*batch_size,  name=name+'_batch_queue')
            return batchA, batchB, batchC, batchD, num_data
        elif mode == 'test':
            batchA, batchB, batchC, batchD, name_batch = tf.train.batch([imgA, imgB, imgC, imgD, filename], batch_size=batch_size, num_threads = 1, capacity=2*batch_size, name=name+'_batch_queue')
            return batchA, batchB, batchC, batchD, name_batch, num_data
        else:
            assert False, 'batch mode error'


def batch_inputs_s2(input_dir, s1_input_dir, batch_size, s1_img_size=256, img_size=256, name='', channelA = 3, channelB = 3, channelC = 3, mode='train', flip=False, crop_size=None, s1_pad_size=None, random_crop=False):
    if mode == 'test':
        flip = False
    if (channelA == 1) and (channelB == 1):
        channel = 1
    else:
        channel = 3

    with tf.device('/cpu:0'):
        # load and preprocess images
        image, num_data, filename = load(input_dir, 16*batch_size, img_size, channel, name=name, mode=mode)
        s1_image, s1_num_data, s1_filename = load(s1_input_dir, 16*batch_size, s1_img_size, channelB, name=name, mode=mode, paired=False)
        assert num_data == s1_num_data, '# of data is not matched'
        # preprocessing
        seed = random.randint(0, 2**31 - 1)
        imgA, imgB, imgC = preprocess(image, img_size, channel, flip, crop_size, random_crop )#, paired=True, seed=seed)
        if s1_pad_size != None:
            s1_image = tf.image.resize_image_with_crop_or_pad(s1_image, s1_pad_size, s1_pad_size)
            s1_img = preprocess(s1_image, s1_pad_size, channelB, flip, s1_img_size, random_crop, paired=False, seed=seed)
        else:
            s1_img = preprocess(s1_image, s1_img_size, channelB, flip, None, False, paired=False)#, seed=seed)
        if channel == 3:
            if channelA == 1:
                imgA = tf.image.rgb_to_grayscale(imgA)
            if channelB == 1:
                imgB = tf.image.rgb_to_grayscale(imgB)
            if channelC == 1:
                imgC = tf.image.rgb_to_grayscale(imgC)
        # make batch
        if mode == 'train':
            batchA, batchB, batchC , s1_batch = tf.train.shuffle_batch([imgA, imgB, imgC, s1_img], batch_size=batch_size, num_threads = 4, capacity=16*batch_size, min_after_dequeue=8*batch_size,  name=name+'_batch_queue')
            return batchA, batchB, batchC, s1_batch, num_data
        elif mode == 'test':
            batchA, batchB, batchC, s1_batch, name_batch = tf.train.batch([imgA, imgB, imgC, s1_img, filename], batch_size=batch_size, num_threads = 1, capacity=2*batch_size, name=name+'_batch_queue')

            return batchA, batchB, batchC, s1_batch, name_batch, num_data
        else:
            assert False, 'batch mode error'


