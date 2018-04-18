#coding:utf-8

import sys, os
import paddle.v2 as paddle

from vgg import vgg_bn_drop
from resnet import resnet_cifar10

with_gpu = os.getenv('WITH_GPU', '0') != "0"

def main():
	# 样本数据的维度
	datadim = 3 * 32 * 32
	# 分类的维度
	classdim = 10

	# PaddlePaddle init
	paddle.init(use_gpu=with_gpu, trainer_count=1)

	image = paddle.layer.data(
		name="image", type=paddle.data_type.dense_vector(datadim))

	# Add neural network config
    # option 1. resnet
    # net = resnet_cifar10(image, depth=32)

    # option 2. vgg
    net = vgg_bn_drop(image)
    # 预测分类输出值
    out = paddle.layer.fc(
    	input=net, size=classdim, act=paddle.activation.Softmax())
    # 真实标签值
    lbl = paddle.layer.data(
    	name="label", type=paddle.data_type.integer_value(classdim))
    # 输出值和真实值之间的损失
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    momentum_optimizer = paddle.optimizer.Momentum(
    	momentum=0.9,
    	regularization=paddle.optimizer.L2Regularization(rate=0.002 * 128),
    	learning_rate=0.1 / 128.0,
    	learning_rate_decay_a=0.1,
    	learning_rate_decay_b=50000 * 100,
    	learning_rate_schedule="discexp")

    # Create trainer
    trainer = paddle.trainer.SGD(
    	cost, parameters=parameters, update_equation=momentum_optimizer)

    # End batch and pass event handler
    def event_handler(event):
    	if isinstance(event, paddle.event.EndIteration):
    		if event.batch_id % 100 == 0:
    			print "\n Pass %d, Batch %d, Cost %f, %s" %(event.pass_id,
    				event.batch_id, event.cost, event.metrics)
    		else:
    			sys.stdout.write('.')
    			sys.stdout.flush()

    	if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(
                reader=paddle.batch(
                    paddle.dataset.cifar.test10(), batch_size=128),
                feeding={'image': 0,
                         'label': 1})
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)		
    			
    # Save the inference topology to protobuf.
    inference_topology = paddle.topology.Topology(layers=out)
    with open("inference_topology.pkl", "wb") as f:
    	inference_topology.serialize_for_inference(f)

    trainer.train(
    reader=paddle.batch(
    	paddle.reader.shuffle(
    		paddle.dataset.cifar.train10(), buf_size=50000),
    	batch_size=128),
    num_passes=200,
    event_handler=event_handler,
    feeding={'image':0, 
    		 'label':1})	

    # inference

    from PIL import image
    import numpy as np
    import os

    def load_image(file):
    	im = Image.open(file)
    	im = im.resize((32, 32), Image.ANTIALIAS)
    	im = np.array(im).astype(np.float32)
    	# The storage order of the loaded image is W(widht),
        # H(height), C(channel). PaddlePaddle requires
        # the CHW order, so transpose them.
        im = im.transpose((2, 0, 1)) # CHW
        # In the training phase, the channel order of CIFAR
        # image is B(Blue), G(green), R(Red). But PIL open
        # image in RGB mode. It must swap the channel order.
        im = im[(2, 1, 0), :, :] # BGR
        im = im.flatten()
        im = im / 255.0
        return im

    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))    
    test_data.append((load_image(cur_dir + "/image/dog.png")))

    # users can remove the comments and change the model name
    # with open('params_pass_50.tar', 'r') as f:
    #    parameters = paddle.parameters.Parameters.from_tar(f)

    probs = paddle.infer(
    	output_layer=out, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)
    print "Label of image/dog.png is: %d" % lab[0][0]

if __name__ == '__main__':
    	main()    



