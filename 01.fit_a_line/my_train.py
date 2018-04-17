with_gpu = os.getenv('WITH_GPU', '0') != '0'

#print with_gpu
#print os.getenv('WITH_GPU', '0')
#print os.getenv('WITH_GPU', '1')

def main():

    # init
    paddle.init(use_gpu=with_gpu, traniner_count=1)
    
    # network config
    # get x from  sample data qian 13 are the abb or featrue of the x
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
    # use linear regression activation to get the y_predict
    y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
    # caculate the cost between real y and y_predicet MSE
    cost = paddle.layer.square_error_cost(input=y_predict, label=y)

    # save the inference topology to protobuf
    inference_topology = paddle.topology.Topology(layers=y_predict)
    with open("inference_topology.pkl", 'wb') as f:
        inference_topology.serialize_for_inference(f)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0)

    trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=optimizer)

    feeding = {'x': 0, 'y': 1}

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
	    if event.batch_id % 100 == 0:
		print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event_batch_id, event.cost)

	if isinstance(event, paddle.event.EndPass):
	   if event.pass_id % 10 == 0:	
