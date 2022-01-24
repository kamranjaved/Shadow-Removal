#used 2 encoders (one with gated convolution and one with normal convolution)
#other settings are same as of 2Disc upadate
#sperated bottle neck
#train model as i2i and inp after 50 epochs
#inverse mask for partial convolution
#similarity lambda 120
#skip connections are between partial encoder and decoder


import numpy as np
import time
import os
import tensorflow as tf
#import numpy as np

#import time
#import os
#import tensorflow as tf
import model_SE_enc_dec_gated_partial_update_econv_skip as model
import input_data2 as input_data
#--------------------------------original domain code----------
#import Encoder_decoder
from utils import *
from flip_gradient import flip_gradient
import pickle as pkl
#from sklearn.manifold import TSNE
#---------------------------------------------------------
from utils_mic import *  ##tf_ssim function defined here
from buffer import *
import vgg19    ###### For VGG   ######

# model gen_config로 바꾸기
#L1+SSIM

class pix2pix(object):
    def __init__(self):
        # input batch
        self.image_size = 272
        self.dimA = 3
        self.dimB = 3
        self.dimC = 3   #mask
        self.dimSC = 3   #mask
        self.tr_dir = "./Video_Test_mic_10"
        self.ts_dir = "./Video_Test_mic_10"
        self.batch_size = 10
        self.flip = True
        self.crop_size = 256    # None or same with image_size, if you don't want to crop
        self.random_crop = True
        # network architecture
        self.ngf = 64
        self.ndf = 64
        self.g_model = 'unetAC'    # Select generator model
        self.n_layer = 5    # number of layers(or blocks)
        self.n_dropout = 3
        self.d_model = 'pixel70'
        self.n_layer_d = 3
        self.norm = 'instance'
        # loss
        self.gan_loss = 'log'
        self.similarity_loss_style = 'STYLE'
        self.similarity_loss = 'VGG_L1_SSIM_TV'
        #self.similarity_loss = 'L1_hole'
        self.similarity_lambda = 120
        # historical batch
        self.history = False
        self.history_batch_size = self.batch_size
        self.buffer_size = 50*self.batch_size
        self.history_start_epoch = 2    # epoch, > 1
        # trainer
        self.max_epoch = 600
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.weight_decay = None
        # training updation
        self.Gen_epoch = 50
        self.I2I_epoch = 500
        # learning rate decay
        self.lr_decay = None
        self.lr_step = None      
        self.step = tf.Variable(0, trainable=False, name='global_step')
        # records
        self.log_dir = "Parameters"#"record"
        self.summary_step = 100   # write tensorboard summary
        self.print_epoch = 1    # print losses
        self.display_epoch = 1    # save test result images
        self.save_epoch = 5   # save model
        # to continue training, you must use same network architecture
        self.continue_training = False
        self.load_dir = "Mix_CP_econv_skip_partial/ckpt"
        self.load_epoch = 250
        self.load_step = 1054250 # It is required in case # of training data is changed
        # Translation function translate images in ts_dir using model on load_dir, load_step(or epoch)
        # you must use same network architecture with training
        # results will be saved on following dir
        self.result_dir = "Trans_Parameters"

        # Select GPU
        self.gpu_num = 1



    def build_trainer(self):
        #vgg = vgg19.Vgg19()   ###### For VGG   ######
        self.original_image_size = self.image_size
        if (self.crop_size != None) and (self.crop_size != self.image_size):
            self.image_size = self.crop_size
        # placeholders
        self.train_mode = tf.placeholder(tf.bool)
        # historical batch
        if self.history:
            self.pool_fake = Buffer(self.buffer_size, self.history_batch_size, [self.image_size, self.image_size, self.dimA+self.dimB], int(time.time()))
            self.history_fakeAB = tf.placeholder(tf.float32, (self.history_batch_size, self.image_size, self.image_size, self.dimA+self.dimB), name='history_fakeAB')
            #self.history_fakeA, self.history_fakeB = tf.split(self.history_fakeAB, [self.dimA, self.dimB], axis=3)
            self.use_history_fake = tf.placeholder(tf.bool)

        # Real images
        tr_realA, tr_realB, tr_realC, self.num_tr = input_data.batch_inputs(self.tr_dir, self.batch_size, img_size=self.original_image_size, name='real_tr', channelA = self.dimA, channelB = self.dimB, channelC = self.dimC, mode='train', flip=self.flip, crop_size=self.crop_size, random_crop=self.random_crop)
        ts_realA, ts_realB, ts_realC,  self.name, self.num_ts = input_data.batch_inputs(self.ts_dir, self.batch_size, img_size=self.original_image_size, crop_size=self.crop_size, name='real_ts', channelA = self.dimA, channelB = self.dimB, channelC = self.dimC, mode='test')
        self.realA = tf.cond(self.train_mode, lambda: tr_realA, lambda: ts_realA)
        self.realB = tf.cond(self.train_mode, lambda: tr_realB, lambda: ts_realB)
        self.realC = tf.cond(self.train_mode, lambda: tr_realC, lambda: ts_realC)
        self.realSC = tf.cond(self.train_mode, lambda: tr_realC, lambda: ts_realC)
        #rA_summ = tf.summary.image('real_imageA', self.realA, max_outputs=3)
        #rB_summ = tf.summary.image('real_imageB', self.realB, max_outputs=3)

        print(tr_realA.get_shape())
        print(tr_realB.get_shape())
        print(tr_realC.get_shape())
        #print(tr_realSC.get_shape())

#----------convert wide mask to binary-----------------------------------------        
        self.realC = tf.add(self.realC, 1)  # our image value is between -1 and 1 so it will add 1 and image values will become 0 and 2
        self.realC = tf.divide(self.realC, 2) # divide by 2 it will 0/2 and 2/2 (0-1)
        self.realC = tf.round(self.realC)#tf.add(self.realC, 0.5)  # round will make value less then 0.5--> 0 and greater than 0.5--> 1 means black and white image
        #self.realC = tf.to_int32(self.realC)
        #self.realC = tf.to_float(self.realC)
        self.realAC = tf.concat([self.realA, self.realC], axis = 3)   #tf.multiply(self.realA, self.realC) 
        print("input of the generator shape", np.shape(self.realAC))

#----------convert Small mask to binary-----------------------------------------        
        self.realSC = tf.add(self.realSC, 1)  # our image value is between -1 and 1 so it will add 1 and image values will become 0 and 2
        self.realSC = tf.divide(self.realSC, 2) # divide by 2 it will 0/2 and 2/2 (0-1)
        self.realSC = tf.round(self.realSC)#tf.add(self.realSC, 0.5)  # round will make value less then 0.5--> 0 and greater than 0.5--> 1 means black and white image
        self.realSAC = tf.concat([self.realA, self.realSC], axis = 3)   #input for the local discriminator with only small mask part generation

#----------create a bidirectional mask for generator--------------------------
        self.realBC = tf.ones(tf.shape(self.realC))
        self.realBC_realC = tf.subtract(self.realBC, self.realC)
        self.B_mask = tf.multiply(self.realA, self.realBC_realC)
        self.realAC_Bmask = tf.concat([self.realA, self.realC, self.B_mask], axis = 3) #concatenate bi directional mask

               
#------------------------------make inverse mask-------------------------------------------------------
        self.B = tf.ones(tf.shape(self.realC))      #    B = [1 1 1]
        self.C = tf.subtract(self.B, self.realC)  #inverse mask

#---------------------------------------------------------------------


        # Generator
        # Select generator model
        # unet
        if self.g_model == 'unetAC':
            #generator = model.unet
            generator_enc = model.unet_encoder
            generator_enc_gated = model.unet_encoder_gated
            generator_enc_partial = model.unet_encoder_partial
            generator_enc_bottleneck = model.unet_bottleneck
            generator_dec = model.unet_decoder_gated   
            self.n_layer = 2
        elif self.g_model == 'unet256':
            generator = model.unet
            self.n_layer = 5
        elif self.g_model == 'unet128':
            generator = model.unet
            self.n_layer = 4
        elif self.g_model == 'unet64':
            generator = model.unet
            self.n_layer = 3
        # resnet
        elif self.g_model == 'resnet':
            generator = model.resnet
        elif self.g_model == 'resnet_6blocks':
            generator = model.resnet
            self.n_layer = 6
        elif self.g_model == 'resnet_9blocks':
            generator = model.resnet
            self.n_layer = 9
        # densenet
        elif self.g_model == 'densenet_4blocks':
            generator = model.densenet
        else:
            assert False, 'generator model error'

#------------------------------------------------------------------------------------


        # generate fake images
        if self.g_model == 'densenet_4blocks': 
            self.gen1 = model.densenet(self.realAC, self.dimB, 4, ngf=16, growth_rate=16, dense_block_size=4, bottleneck=True, norm='instance', name='Gen_', train_mode=True)
        else:
            _,_,_,_,self.genlasts, _ = generator_enc(self.realAC, self.dimB, self.n_layer , self.ngf, 4, self.norm, 'Gen_', self.train_mode, n_dropout=self.n_dropout)
            self.gen0s,self.gen1s,self.gen2s,self.gen3s,self.genlt,self.enc_layers = generator_enc_partial(self.realA, self.C, self.dimB, self.n_layer , self.ngf, 4, self.norm, 'Gen_p', self.train_mode, reuse=False, n_dropout=self.n_dropout) #encd target data output
            #_,_,_,_,self.genlgt,_ = generator_enc_gated(self.realAC, self.dimB, self.n_layer , self.ngf, 4, self.norm, 'Geng_', self.train_mode, reuse=False, n_dropout=self.n_dropout) #encd target data output

            #print('encoder output for source data0', self.gen0s.get_shape())
            #print('encoder output for source data1', self.gen1s.get_shape())
            #print('encoder output for source data2', self.gen2s.get_shape())
            #print('encoder output for source data3', self.gen3s.get_shape())
            #print('last layer of encoder output for source data', self.genlasts.get_shape()) # output of the last layer of the encoder using source data
            #print('encoder total layers',self.enc_layers)

            #print('encoder output for partial last layer', self.genlt.get_shape()) # output of the last layer of the encoder using target data
            #print('encoder output for gated last layer', self.genlgt.get_shape()) # output of the last layer of the encoder using target data
            self.dec_input = tf.concat([self.genlasts, self.genlt], axis=3)
            print('encoder output after concatenation gated and normal encoder', self.dec_input.get_shape()) # output of the last layer of the encoder using target data

#-------------bottle neck layers----------------------------------

            self.b1,self.b2,self.b3,self.b4,self.b5,self.b6,self.b7,self.genbt,_ = generator_enc_bottleneck(self.dec_input, 4, self.norm, 'Gen_b', self.train_mode, reuse = False) #encd target data outpub

#-----------------------------------------------------------------

            self.gen = generator_dec(self.genbt, self.gen0s,self.gen1s,self.gen2s,self.gen3s, self.genlt,self.b1,self.b2,self.b3,self.b4,self.b5,self.b6,self.b7,self.genbt, 13, self.dimB, self.n_layer , self.ngf, 4, self.norm, 'Gen_d', self.train_mode, n_dropout=self.n_dropout ) # i replaced self.encoder_layer with 13

            #print("output of the generator shape is", np.shape(self.gen))

            '''
#--------------------this part is only for local discriminator--------------------------
            self.A = tf.multiply(self.gen, self.realC) #    A = Ig*m
            self.B = tf.ones(tf.shape(self.realC))      #    B = [1 1 1]
            self.C = tf.subtract(self.B, self.realC)    #    C = B - Mask
            self.D = tf.multiply(self.realA, self.C)
            self.gen_l = tf.add(self.A, self.D)  # mask part will be of the generated one and the remaining part will be of real one
            print("output of the replaced with orinial image", np.shape(self.gen_l))            
#--------------------------------------------------------------------------------------

#--------------------this part is only for if want to inpaint with samll mask--------------------------
            self.SA = tf.multiply(self.gen_l, self.realSC) #    A = Ig*m-->SC is small mask
            self.SB = tf.ones(tf.shape(self.realSC))      #    B = [1 1 1]
            self.SC = tf.subtract(self.SB, self.realSC)    #    C = B - Mask
            self.SD = tf.multiply(self.gen_l, self.SC)
            self.gen_ls = tf.add(self.SA, self.SD)  # mask part will be of the  wide generated one+smaller one and the remaining part will be of real one
            print("output of the replaced with orinial image", np.shape(self.gen_ls))            
#--------------------------------------------------------------------------------------
            '''

#----------------------------if we use only small mask--------------------------------

#--------------------this part is only for if want to inpaint with samll mask--------------------------
            self.SA = tf.multiply(self.gen, self.realSC) #    A = Ig*m-->SC is small mask
            self.SB = tf.ones(tf.shape(self.realSC))      #    B = [1 1 1]
            self.SC = tf.subtract(self.SB, self.realSC)    #    C = B - Mask
            self.SD = tf.multiply(self.realA, self.SC)
            self.gen_ls = tf.add(self.SA, self.SD)  # mask part will be of the  wide generated one+smaller one and the remaining part will be of real one
            #print("output of the replaced with orinial image", np.shape(self.gen_ls))            
#--------------------------------------------------------------------------------------




        g_summ = tf.summary.image('generated_image', self.gen, max_outputs=3)

        # Global discriminator inputs
        self.genAB = tf.concat([self.realAC, self.gen], axis=3)
        self.realAB = tf.concat([self.realAC, self.realB], axis=3)

        # Local discriminator inputs
        self.genAB_l = tf.concat([self.realAC, self.gen_ls], axis=3)
        self.realAB_l = tf.concat([self.realAC, self.realB], axis=3)


        if self.history:
            self.d_inp_fake = tf.cond(self.use_history_fake, lambda: tf.concat([self.genAB, self.history_fakeAB], axis=0), lambda: self.genAB) # for global dis
            self.d_inp_fake_l = tf.cond(self.use_history_fake, lambda: tf.concat([self.genAB_l, self.history_fakeAB], axis=0), lambda: self.genAB) # for local dis
        else:
            self.d_inp_fake = self.genAB  #for global discriminator
            self.d_inp_fake_l = self.genAB_l  #for local discriminator
            #print("disc. input fake shape", np.shape(self.d_inp_fake))
        self.d_inp_real = self.realAB  #for global discriminator
        self.d_inp_real_l = self.realAB  #for local discriminator
        #print("disc. inp real shape", np.shape(self.d_inp_real))

        # Discriminator
        if self.d_model == 'pixel':
            discriminator = model.pixel
        elif self.d_model == 'pixel16':
            discriminator = model.pixel
            self.n_layer_d = 1
        elif self.d_model == 'pixel34':
            discriminator = model.pixel
            self.n_layer_d = 2
        elif self.d_model == 'pixel70':
            discriminator = model.pixel
            self.n_layer_d = 3
        elif self.d_model == 'pixel142':
            discriminator = model.pixel
            self.n_layer_d = 4
        elif self.d_model == 'pixel286':
            discriminator = model.pixel
            self.n_layer_d = 5
        else:
            assert False, 'discriminator model error'
       
        if self.gan_loss == 'hinge':
            self.sigmoid = False
        else:
            self.sigmoid = True

#--------------------for global discriminator--------------------------------------------------------------------------------------------       
        self.dreal = discriminator(self.d_inp_real, self.n_layer_d, self.ndf, 4, self.norm, 'Dis_', self.train_mode, reuse=False)#, sigmoid_output=self.sigmoid)
        self.dfake = discriminator(self.d_inp_fake, self.n_layer_d, self.ndf, 4, self.norm, 'Dis_', self.train_mode, reuse=True)#, sigmoid_output=self.sigmoid)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------for local discriminator--------------------------------------------------------------------------------------------       
        self.dreal_l = discriminator(self.d_inp_real_l, self.n_layer_d, self.ndf, 4, self.norm, 'Disl_', self.train_mode, reuse=False)#, sigmoid_output=self.sigmoid)
        self.dfake_l = discriminator(self.d_inp_fake_l, self.n_layer_d, self.ndf, 4, self.norm, 'Disl_', self.train_mode, reuse=True)#, sigmoid_output=self.sigmoid)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------for global discriminator-------------------------------------------------------------------------------------------- 
        # adversarial loss
        self.g_loss_g, self.d_loss_g = model.GAN_loss(self.dfake, self.dreal, self.gan_loss)  
#---------------------------------------------------------------------------------------------------------------------------------------- 
#--------------------for local discriminator-------------------------------------------------------------------------------------------- 
        # adversarial loss
        self.g_loss_l, self.d_loss_l = model.GAN_loss(self.dfake_l, self.dreal_l, self.gan_loss)  
#---------------------------------------------------------------------------------------------------------------------------------------- 
        self.g_loss = self.g_loss_g + self.g_loss_l
        self.d_loss = 0.3*self.d_loss_g + 0.7*self.d_loss_l


#--------------------for global discriminator-------------------------------------------------------------------------------------------- 
        # global discriminator outputs of generated images
        self.gen_score = tf.reduce_mean(self.dfake, axis=[1,2,3])
        self.real_score = tf.reduce_mean(self.dreal, axis=[1,2,3])

#--------------------for local discriminator-------------------------------------------------------------------------------------------- 
        # local discriminator outputs of generated images
        self.gen_score_l = tf.reduce_mean(self.dfake_l, axis=[1,2,3])
        self.real_score_l = tf.reduce_mean(self.dreal_l, axis=[1,2,3])


           # similarity loss
        self.sgen = self.gen
        self.srealB = self.realB

#----------------for global and local discriminator---------------------------------
        #self.sgen = tf.concat([self.gen_male, self.gen_female], axis=0)
        #self.srealB = tf.concat([self.realB_male, self.realB_female], axis=0)

        vgg_r = vgg19.Vgg19()
        with tf.name_scope("Real_vgg"):
             vgg_r.build(self.srealB)
        self.feature_real3 = vgg_r.conv3_1
        
        vgg_g = vgg19.Vgg19()
        with tf.name_scope("G_vgg"):
             vgg_g.build(self.sgen)
        self.feature_gen3 =  vgg_g.conv3_1

        self.feature_real4 = vgg_r.conv4_1
        
        self.feature_gen4 =  vgg_g.conv4_1

        self.feature_real5 = vgg_r.conv5_1

        self.feature_gen5 =  vgg_g.conv5_1


        
        if self.similarity_loss_style == 'STYLE': #from edge connect paper using gram matrix---> gram_matrix defined in utils_mic 
            self.style_loss = tf.reduce_mean(tf.subtract(gram_matrix(self.feature_gen3), gram_matrix(self.feature_real3))) +tf.reduce_mean(tf.subtract(gram_matrix(self.feature_gen4), gram_matrix(self.feature_real4))) + tf.reduce_mean(tf.subtract(gram_matrix(self.feature_gen5), gram_matrix(self.feature_real5)))  # tf.reduce_mean(tf.abs(self.sgen - self.srealB))   ###### For VGG   ######


        if self.similarity_loss == 'VGG':
            self.s_loss = tf.reduce_mean(tf.subtract(self.feature_gen3, self.feature_real3) ** 2) +tf.reduce_mean(tf.subtract(self.feature_gen4, self.feature_real4) ** 2) + tf.reduce_mean(tf.subtract(self.feature_gen5, self.feature_real5) ** 2)  # tf.reduce_mean(tf.abs(self.sgen - self.srealB))   ###### For VGG   ######
            print('Shape of s_loss ', np.shape(self.s_loss))

        elif self.similarity_loss == 'VGG_L1_SSIM':
            self.s_loss =  tf.reduce_mean(tf.abs(self.sgen - self.srealB)) + (1 - tf_ssim(self.sgen, self.srealB, size=11, sigma=1.5)) + tf.reduce_mean(tf.subtract(self.feature_gen3, self.feature_real3) ** 2) +tf.reduce_mean(tf.subtract(self.feature_gen4, self.feature_real4) ** 2) + tf.reduce_mean(tf.subtract(self.feature_gen5, self.feature_real5) ** 2)   # tf.reduce_mean(tf.abs(self.sgen - self.srealB))   ###### For VGG   ######

        elif self.similarity_loss == 'L1_valid':
            self.s_loss = tf.reduce_mean(tf.abs(self.realC*(self.sgen - self.srealB)))
        elif self.similarity_loss == 'L1_hole':
            self.s_loss = tf.reduce_mean(tf.abs(self.C*(self.sgen - self.srealB)))


        elif self.similarity_loss == 'L1':
            self.s_loss = tf.reduce_mean(tf.abs(self.sgen - self.srealB))
        elif self.similarity_loss == 'L2':
            self.s_loss = tf.reduce_mean((self.sgen - self.srealB)**2)
        elif self.similarity_loss == 'ssim':
            self.s_loss = 1 - tf_ssim(self.sgen, self.srealB, size=11, sigma=1.5)
        elif (self.similarity_loss == 'L1_ssim') or (self.similarity_loss == 'ssim_L1'):
            self.s_loss = 1.5*tf.reduce_mean(tf.abs(self.sgen - self.srealB)) + (1 - tf_ssim(self.sgen, self.srealB, size=11, sigma=1.5))
        elif self.similarity_loss == 'VGG_L1_SSIM_TV':
            self.s_loss =  tf.reduce_mean(tf.abs(self.sgen - self.srealB)) + (1 - tf_ssim(self.sgen, self.srealB, size=11, sigma=1.5)) + 0.33*loss_normalize(tf.reduce_mean(tf.subtract(self.feature_gen3, self.feature_real3) ** 2)) + 0.33*loss_normalize(tf.reduce_mean(tf.subtract(self.feature_gen4, self.feature_real4) ** 2)) + 0.33*loss_normalize(tf.reduce_mean(tf.subtract(self.feature_gen5, self.feature_real5) ** 2))  +0.1*loss_normalize(tf.reduce_mean(tf.image.total_variation(self.sgen)))

        elif self.similarity_loss == 'VGG_L1_SSIM_TV_STYLE':
            self.s_loss =  tf.reduce_mean(tf.abs(self.sgen - self.srealB)) + (1 - tf_ssim(self.sgen, self.srealB, size=11, sigma=1.5)) + 0.33*loss_normalize(tf.reduce_mean(tf.subtract(self.feature_gen3, self.feature_real3) ** 2)) + 0.33*loss_normalize(tf.reduce_mean(tf.subtract(self.feature_gen4, self.feature_real4) ** 2)) + 0.33*loss_normalize(tf.reduce_mean(tf.subtract(self.feature_gen5, self.feature_real5) ** 2))  +0.1*loss_normalize(tf.reduce_mean(tf.image.total_variation(self.sgen))) + self.style_loss

        elif self.similarity_loss == 'VGG_TV':
            self.s_loss =  tf.reduce_mean(tf.subtract(self.feature_gen3, self.feature_real3) ** 2) +tf.reduce_mean(tf.subtract(self.feature_gen4, self.feature_real4) ** 2) + tf.reduce_mean(tf.subtract(self.feature_gen5, self.feature_real5) ** 2)  + 0.1*tf.image.total_variation(self.sgen) 
        else:
            assert False, 'similarity loss error'
        self.g_loss =   tf.add(self.g_loss, self.similarity_lambda*self.s_loss)

        # variable list
        t_vars = tf.trainable_variables()
        self.Gen_vars = [var for var in t_vars if 'Gen_' in var.name]
        self.Dis_vars = [var for var in t_vars if 'Dis_' in var.name]
        self.Disl_vars = [var for var in t_vars if 'Disl_' in var.name]

        for variable in self.Gen_vars:
            shape = variable.get_shape()
            #print ("generator variable and their shape", variable.name, shape) 
        for variable in self.Dis_vars:
            shape = variable.get_shape()
            #print ("discriminator variable and their shape", variable.name, shape)    
   

        # kernel norms
        #self.weight_names = [var.name for var in t_vars if "weights" in var.name]
        # It works because no linear in current models
        #self.kernel_norms = [tf.reduce_sum(var**2, axis=[0,1,2]) for var in t_vars if "weights" in var.name]

        # learning rate
        if self.lr_decay == None:
            self.lr = self.learning_rate
        else:
            self.lr = tf.train.exponential_decay(self.learning_rate, self.step,self.lr_step, self.lr_decay, staircase=True)
        # Optimizer
        Gen_optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2, name='Gen_Adam')
        Dis_optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2, name='Dis_Adam')
        Disl_optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2, name='Disl_Adam')

        # Compute gradiants
        Gen_grad = Gen_optimizer.compute_gradients(self.g_loss, var_list=self.Gen_vars)
        Dis_grad = Dis_optimizer.compute_gradients(self.d_loss_g, var_list=self.Dis_vars)
        Disl_grad = Dis_optimizer.compute_gradients(self.d_loss_l, var_list=self.Disl_vars)

        # Updates
        self.g_optim = Gen_optimizer.apply_gradients(Gen_grad, global_step=self.step)
        self.d_optim = Dis_optimizer.apply_gradients(Dis_grad)
        self.dl_optim = Dis_optimizer.apply_gradients(Disl_grad)

        return





    def train(self):
        epoch = int(self.num_tr/self.batch_size)
        ts_epoch = int(self.num_ts/self.batch_size)
        print_step = int(epoch*self.print_epoch) 
        display_step = int(epoch*self.display_epoch)

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.log_dir+'/ckpt'):
            os.mkdir(self.log_dir+'/ckpt')
        if not os.path.exists(self.log_dir+'/image'):
            os.mkdir(self.log_dir+'/image')
        if not os.path.exists(self.log_dir+'/summary'):
            os.mkdir(self.log_dir+'/summary')

        # initializer
        init_op = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(max_to_keep=None)
        # Summary
        self.merged_summary = tf.summary.merge_all()

        # feed_dict
        feed_dict = {self.train_mode: True}
        feed_dict_ts = {self.train_mode: False}
        # fetch dict
        tr_dict = {'d_optim': self.d_optim, 'dl_optim': self.dl_optim, 'g_optim': self.g_optim, 'd_loss': self.d_loss, 'g_loss': self.g_loss} #for joint training gen+2Disc
        tr_dict2 = {'g_optim': self.g_optim, 'd_loss': self.d_loss, 'g_loss': self.g_loss} # for Generator training only
        #tr_dict2 = {'g_optim': self.g_optim, 'g_loss': self.g_loss} # for Generator training only
        tr_dict3 = {'d_optim': self.d_optim, 'g_optim': self.g_optim, 'd_loss': self.d_loss_g, 'g_loss': self.g_loss_g} # for Generator and I2I discriminator only training only

        ts_dict = {'inp': self.realA, 'gen': self.gen, 'target': self.realB, 'real_score': self.real_score, 'gen_score': self.gen_score, 'name': self.name, 'd_loss': self.d_loss, 'g_loss': self.g_loss, 's_loss': self.s_loss}#, 'kernel_norm': self.kernel_norms}

        if self.history:
            feed_dict.update({self.use_history_fake: False, self.history_fakeAB: np.zeros([self.history_batch_size, self.image_size, self.image_size, self.dimA+self.dimB])})
            feed_dict_ts.update({self.use_history_fake: False, self.history_fakeAB: np.zeros([self.history_batch_size, self.image_size, self.image_size, self.dimA+self.dimB])})
            tr_dict.update({'gen_img': self.genAB})

        # ConfigProto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
        config.log_device_placement=False
        # Training
        with tf.Session(config=config) as sess:
            print("Start session")
            summary_writer = tf.summary.FileWriter(self.log_dir+'/summary', sess.graph)
            if self.continue_training == True:
                if self.load_step == None:
                    self.load_step = self.load_epoch * epoch
                epoch_i = self.load_epoch
                self.saver.restore(sess, self.load_dir+"/pix2pix-"+str(self.load_step))
                print("Model restored from epoch %d." % self.load_epoch)
            else:
                epoch_i = 0
                sess.run(init_op)
                print("Initialization done")

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")

            # Do training
            print("Start training")
            for i in range(epoch*epoch_i+1, int(self.max_epoch*epoch)+1):
                # epoch check
                if i % epoch == 1:
                    epoch_i = epoch_i + 1
                    if epoch_i == self.history_start_epoch:
                        if self.history:
                            feed_dict.update({self.use_history_fake: True, self.history_fakeAB: self.pool_fake.sample()})

                # Update training 
                if i <= self.Gen_epoch:  
                   tr_result = sess.run(tr_dict2, feed_dict=feed_dict)  # Train generator only for self.Gen_epoch
                elif i > self.Gen_epoch and i <= self.I2I_epoch:  
                   tr_result = sess.run(tr_dict, feed_dict=feed_dict)   # Train generator  and I2I only from self.Gen_epoch to self.I2I_epoch
                else:
                   tr_result = sess.run(tr_dict, feed_dict=feed_dict)
                assert not np.isnan(tr_result['d_loss']), 'Model diverged with d_loss = NaN'
                assert not np.isnan(tr_result['g_loss']), 'Model diverged with g_loss = NaN'
                # history_fake 
                if self.history:
                    self.pool_fake.push(tr_result['gen_img'])
                    if feed_dict[self.use_history_fake]:
                        feed_dict.update({self.history_fakeAB: self.pool_fake.sample()})

                #records
                if i % self.summary_step == 0:
                    summary_str = sess.run(self.merged_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
                if i % print_step == 0:
                    cur_time = time.localtime(time.time())
                    print("%d.%d.%d %d:%d:%d||epoch %d. d_loss: %.5f g_loss: %.5f" % (cur_time.tm_year, cur_time.tm_mon, cur_time.tm_mday, cur_time.tm_hour, cur_time.tm_min, cur_time.tm_sec, epoch_i, tr_result['d_loss'], tr_result['g_loss']))
                # visualization
                if i % display_step == 0:
                    vis_dir = self.log_dir+'/image/ep'+str(epoch_i)+'_iter'+str(i)
                    gen_score = []
                    real_score = []
                    for j in range(ts_epoch):
                        ts_result = sess.run(ts_dict, feed_dict=feed_dict_ts)
                        visualize_results(vis_dir, ts_result)
                        gen_score = np.concatenate([gen_score, ts_result['gen_score']], axis=0)
                        real_score = np.concatenate([real_score, ts_result['real_score']], axis=0)
                    write_html(vis_dir, label='score', output_labels=gen_score, target_labels=real_score)
                    #write_txt(vis_dir, ts_result, self.weight_names, epoch_i)
                # save model
                if i % (epoch*self.save_epoch) == 0:
                    save_path = self.saver.save(sess, self.log_dir+'/ckpt/pix2pix', global_step=i)
                    print("Model saved in file: %s" % save_path)
    
            summary_str = sess.run(self.merged_summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            save_path = self.saver.save(sess, self.log_dir+'/ckpt/pix2pix', global_step=i)
            print("Model saved in file: %s" % save_path)

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)
            sess.close()
        
        return


    def translate_images(self):
        epoch = int(self.num_tr/self.batch_size)
        ts_epoch = int(self.num_ts/self.batch_size)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        # Add ops to save and restore all the variables.
        self.loader = tf.train.Saver(max_to_keep=None)

        # feed_dict
        feed_dict = {self.train_mode: False}
        # fetch dict
        fetch_dict = {'inp': self.realA, 'gen': self.gen, 'target': self.realB, 'real_score': self.real_score, 'gen_score': self.gen_score, 'name': self.name, 'd_loss': self.d_loss, 'g_loss': self.g_loss, 's_loss': self.s_loss}#, 'kernel_norm': self.kernel_norms}

        if self.history:
            feed_dict.update({self.use_history_fake: False, self.history_fakeAB: np.zeros([self.history_batch_size, self.image_size, self.image_size, self.dimA+self.dimB])})
 
        # ConfigProto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
        config.log_device_placement=False
        with tf.Session(config=config) as sess:
            print("Start session")
            if self.load_step == None:
                self.load_step = self.load_epoch * epoch
            epoch_i = self.load_epoch
            self.loader.restore(sess, self.load_dir+"/pix2pix-"+str(self.load_step))
            print("Model restored from epoch %d." % self.load_epoch)

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")

            # Translation
            print("Translation")
            vis_dir = self.result_dir+'/ep'+str(epoch_i)
            gen_score = []
            real_score = []
            for j in range(ts_epoch):
                ts_result = sess.run(fetch_dict, feed_dict=feed_dict)
                visualize_results(vis_dir, ts_result)
                gen_score = np.concatenate([gen_score, ts_result['gen_score']], axis=0)
                real_score = np.concatenate([real_score, ts_result['real_score']], axis=0)
            write_html(vis_dir, label='score', output_labels=gen_score, target_labels=real_score)
            #write_txt(vis_dir, ts_result, self.weight_names, epoch_i)
            print("Done")
            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)
            sess.close()
        return

