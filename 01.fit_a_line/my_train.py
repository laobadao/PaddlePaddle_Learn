#coding:utf-8

import os
import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing

with_gpu = os.getenv('WITH_GPU', '0') != '0'

#print with_gpu
#print os.getenv('WITH_GPU', '0')
#print os.getenv('WITH_GPU', '1')

def main():

    # init
    paddle.init(use_gpu=with_gpu, traniner_count=1)
    
    # network config
    # get x from  
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
    y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
    # caculate the cost between real y and y_predicet
    
