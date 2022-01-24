import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os

import pix2pixB_CP_econv_skip
from utils import *
from utils_mic import *

# Build
net = pix2pixB_CP_econv_skip.pix2pix()
if net.gpu_num != None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(net.gpu_num)
print(device_lib.list_local_devices())

net.build_trainer()

# Variables
print("Gen_vars")
for i in range(len(net.Gen_vars)):
    print(net.Gen_vars[i].name)
print("Dis_vars")
for i in range(len(net.Dis_vars)):
    print(net.Dis_vars[i].name)
print("build model finished")
print("Tr_num: "+str(net.num_tr))
print("Ts_num: "+str(net.num_ts))

# Train
net.translate_images()
