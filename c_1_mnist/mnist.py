# coding:utf-8

import os
from PIL import Image
import numpy as np
import paddle.v2 as paddle

# use gpu , 0=no , 1=yes

with_gpu = os.getenv('WITH_GPU', '0') != '1'

# define neural network arch

def convolutional_neural_network_org(img):
    # first cnn
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
		input=img,
		filter_zise=5,
		num_filters=20,
		num_channel=1,
		pool_size=2,
		pool_stride=2,
		act=paddle.activation.Relu())	
    # second cnn
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
		input=conv_pool_1,
		filter_size=5,
		num_filters=50,
		num_channel=20,
		pool_size=2,
		pool_stride=2,
		act=paddle.activation().Relu())
    # fc 
    predict= padle.layer.fc(
	input=conv_pool_2, size=10, act=paddle.activation.Softmax())

    return predict

def main():

    # init device
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # read data
    images = paddle.layer.data(
	name='pixel',type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
	name='label', type=paddle.data_type.integer_value(10))

    # call nn arch defined before
    predict = convolutional_neural_network_org(images)
    
    # define cost function
    


