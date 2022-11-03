from new_dm import Data_Factory
from parallel_sampler import Para_Sampler
from model import Text_encoder,doc_label_metric,Train
import numpy as np


import click
@click.command()
@click.option('--data', default='Wiki10-31K')
@click.option('--la', default=1.0)
@click.option('--ba', default=1.0)
@click.option('--emb_num', default=256)
@click.option('--n_negative', default=1000)

def main(data,la,ba,emb_num,n_negative):

	min_rating = 1
	max_df = 0.5
	vocab_size = 40000
	split_ratio = 0.9
	TOP_N=[1,3,5,10]
	path = f'../data/{data}'
	log_path = f'log/{data}.log'
	best_log_path = f'log/{data}_best.log'
	data_Factory = Data_Factory()
	D = data_Factory.preprocess(train_text=f'{path}/train_raw_texts.txt', train_tag=f'{path}/train_labels.txt',
		test_text=f'{path}/test_raw_texts.txt', test_tag= f'{path}/test_labels.txt',_max_df = max_df,_vocab_size=vocab_size)
	
	#data_Factory.save(path,D)
	#D=data_Factory.load(path)

	X_train, X_test = D['X'][:D['train_num']], D['X'][D['train_num']:]
	Y_train, Y_test = D['Y'][:D['train_num']], D['Y'][D['train_num']:]

	

	n_tags = len(D['Y_tag'])
	#sampler = Para_Sampler(Y_train=Y_train, n_tags=n_tags, batch_size=1024, n_negative=500,n_workers=30)

	X_train, X_test, maxlen_doc = data_Factory.pad_sequence(X_train, X_test,512)

	#bl = data_Factory.get_binary_label(Y_train,len(D['Y_tag']))

	word_dim = 300

	# Read Glove word vectors
	pretrain_w2v = f'../glove/glove.6B.{word_dim}d.txt'
	if pretrain_w2v is None:
	    init_WV = None
	else:
	    init_WV = data_Factory.read_pretrained_word2vec(pretrain_w2v, D['X_vocab'], word_dim)

	sampler = Para_Sampler(Y_train=Y_train, n_tags=n_tags, batch_size=1024, n_negative=n_negative,n_workers=30)



	log = open(log_path, 'a')
	log.write("lambda:%.2f,beta:%.2f emb_num:%d n_negative:%d \n\n" % (la,ba,emb_num,n_negative))

	log.flush()
	log.close()


	text_encoder = Text_encoder(init_WV=init_WV, vocab_size=len(D['X_vocab']) + 1, word_dim=word_dim, num_kel=512,emb_num=emb_num)
	l_model = doc_label_metric(n_text=len(X_train),n_tags=n_tags,tag_dim=emb_num,lamba=la,beta=ba)

	train=Train(sampler=sampler,Text_encoder=text_encoder,doc_label_metric=l_model,max_iter=6,n_tags=n_tags,X_train=X_train,X_test=X_test, Y_train=Y_train,Y_test=Y_test)

	best_pre = train.train_loop(TOP_N,log_path)
	best_pre= np.around(best_pre, 4)

	log = open(best_log_path, 'a')
	log.write("lambda:%.2f,beta:%.2f emb_num:%d n_negative:%d \n\n" % (la,ba,emb_num,n_negative))
	log.write(("Precision: %s Recall: %s NDCG: %s HD: %s\n\n" % 
		(best_pre[0], best_pre[1], best_pre[2], best_pre[3])))
	log.flush()
	log.close()

if __name__ == '__main__':
	main()