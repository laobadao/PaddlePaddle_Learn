#coding:utf-8

import paddle.v2 as paddle
import cPickle
import copy
import os

with_gpu = os.getenv('WITH_GPU', '0') != '0'

def get_usr_combined_features():
	# 对 user_id 进行 embedding fc 操作
	uid = paddle.layer.data(
		name='user_id',
		type=paddle.data_type.integer_value(
			paddle.dataset.movielens.max_user_id() + 1))
	usr_emb = paddle.layer.embedding(input=uid, size=32)
	usr_fc = paddle.layer.fc(input=usr_emb, size=32)

	# 对 usr_gender_id 进行 embedding fc 操作
	usr_gender_id = paddle.layer.data(
		name='gender_id',type=paddle.data_type.integer_value(2))
	usr_gender_emb = paddle.layer.embedding(input=usr_gender_id, size=16)
	usr_gender_fc = paddle.layer.fc(input=usr_gender_emb, size=16)	
	
	# 对 usr_age_id 进行 embedding fc 操作
	usr_age_id = paddle.layer.data(
		name='age_id',
		type=paddle.data_type.integer_value(
			len(paddle.dataset.movielens.age_table)))
	usr_age_emb = paddle.layer.embedding(input=usr_age_id, size=16)				
	usr_age_fc = paddle.layer.fc(input=usr_age_emb, size=16)

	# 对 usr_job_id 进行 embedding fc 操作
	usr_job_id = paddle.layer.data(
		name="job_id",
		type=paddle.data_type.integer_value(paddle.dataset.movielens.max_job_id() + 1))
	usr_job_emb = paddle.layer.embedding(input=usr_job_id, size=16)
	usr_job_fc = paddle.layer.fc(input=usr_job_emb, size=16)

	get_usr_combined_features = paddle.layer.fc(
		input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc],
		size=200,
		act=paddle.activation.Tanh())
	return get_usr_combined_features

def get_mov_combined_features():
	movie_title_dict = paddle.dataset.movielens.get_movie_title_dict()
	mov_id = paddle.layer.data(
		name='movie_id',
		type=paddle.data_type.integer_value(
			paddle.dataset.movielens.max_movie_id() + 1))	
	mov_emb = paddle.layer.embedding(input=mov_id, size=32)
	mov_fc = paddle.layer.fc(input=mov_emb, size=32)

	# copy : continue
	mov_categories = paddle.layer.data(
        name='category_id',
        type=paddle.data_type.sparse_binary_vector(
            len(paddle.dataset.movielens.movie_categories())))
        mov_categories_hidden = paddle.layer.fc(input=mov_categories, size=32)

        mov_title_id = paddle.layer.data(
        name='movie_title',
        type=paddle.data_type.integer_value_sequence(len(movie_title_dict)))
        mov_title_emb = paddle.layer.embedding(input=mov_title_id, size=32)
        mov_title_conv = paddle.networks.sequence_conv_pool(
        input=mov_title_emb, hidden_size=32, context_len=3)

        mov_combined_features = paddle.layer.fc(
        input=[mov_fc, mov_categories_hidden, mov_title_conv],
        size=200,
        act=paddle.activation.Tanh())
        return mov_combined_features


def main():
    paddle.init(use_gpu=with_gpu)
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()
    inference = paddle.layer.cos_sim(
        a=usr_combined_features, b=mov_combined_features, size=1, scale=5)
    cost = paddle.layer.square_error_cost(
        input=inference,
        label=paddle.layer.data(
            name='score', type=paddle.data_type.dense_vector(1)))

    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-4))
    feeding = {
        'user_id': 0,
        'gender_id': 1,
        'age_id': 2,
        'job_id': 3,
        'movie_id': 4,
        'category_id': 5,
        'movie_title': 6,
        'score': 7
    }

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d Batch %d Cost %.2f" % (
                    event.pass_id, event.batch_id, event.cost)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.movielens.train(), buf_size=8192),
            batch_size=256),
        event_handler=event_handler,
        feeding=feeding,
        num_passes=1)

    user_id = 234
    movie_id = 345

    user = paddle.dataset.movielens.user_info()[user_id]
    movie = paddle.dataset.movielens.movie_info()[movie_id]

    feature = user.value() + movie.value()

    infer_dict = copy.copy(feeding)
    del infer_dict['score']

    prediction = paddle.infer(
        output_layer=inference,
        parameters=parameters,
        input=[feature],
        feeding=infer_dict)
    
    score = (prediction + 5) / 2
    print score
    print "[Predict] User %d Rating Movie %d With Score %.2f"%(user_id, movie_id, score)


if __name__ == '__main__':
    main()


