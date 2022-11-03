import tensorflow as tf
from tensorflow.keras.layers import Bidirectional,GRU,Embedding,Dense,LSTM,Dropout
import numpy as np
from eval_metrics import ID2ID, AP, NDCG, PrecisionRecall, val_format,HD
from new_dm import Data_Factory
from parallel_sampler import Para_Sampler
import time
from tqdm import tqdm
import tensorflow_addons as tfa

class Attention(tf.keras.layers.Layer):
	
    def __init__(self, units,**kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units=units

    def build(self, input_shape):
        self.w_omega = self.add_weight(name='w_omega',  shape=(input_shape[2], self.units),initializer='uniform', trainable=True)
        self.b_omega = self.add_weight(name='b_omega', shape=(self.units,),initializer='uniform',trainable=True)
        self.u_omega = self.add_weight(name='u_omega', shape=(self.units,),initializer='uniform',trainable=True)
        self.built = True

    def call(self, x):
        v = tf.tanh(tf.tensordot(x, self.w_omega, axes=1) + self.b_omega)
        vu = tf.tensordot(v,self.u_omega, axes=1)
        alphas = tf.nn.softmax(vu)         
        output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), 1)
        return output

class Text_encoder(tf.keras.Model):

	def __init__(self,vocab_size,word_dim,init_WV,num_kel,emb_num):
		super(Text_encoder, self).__init__()

		self.embedding = Embedding(vocab_size, word_dim, weights=[init_WV], trainable=True)
		self.bigru = Bidirectional(LSTM(num_kel, return_sequences=True), merge_mode='sum')
		#self.drop = Dropout(0.5)
		#self.drop_1 = Dropout(0.2)
		self.self_att = Attention(100)
		self.fc = Dense(emb_num,activation='tanh')


	def call(self,inputs,train=True):
		input_emb = self.embedding(inputs)
		'''
		if train:
			input_emb = self.drop_1(input_emb)
			gru_seq=self.bigru(input_emb)
			gru_seq = self.drop(gru_seq)
		else:
			gru_seq=self.bigru(input_emb)
		'''
		
		gru_seq=self.bigru(input_emb)

		output = self.self_att(gru_seq)
		#output = tf.math.l2_normalize(output,axis=2)
		output = self.fc(output)



		return output

class doc_label_metric(tf.keras.Model):

    def __init__(self,n_text,n_tags,tag_dim,lamba,beta):
        super(doc_label_metric, self).__init__()
        
        self.tag_embeddings = tf.Variable(tf.random.normal([n_tags,tag_dim], stddev=0.01),trainable=True)
        self.text_embeddings = tf.Variable(tf.random.normal([n_text,tag_dim], stddev=0.01),trainable=True)
        self.lamba = lamba
        self.beta = beta
        #self.tag_embeddings = tf.math.l2_normalize(self.tag_embeddings,axis=1)
        #self.text_embeddings = tf.math.l2_normalize(self.text_embeddings)
        #self.B = tf.Variable(np.array([1.0]*n_text), dtype=tf.float32,trainable=True)
        #self.B1 = tf.Variable(np.array([1.0]*n_tags),dtype=tf.float32,trainable=True)
    
    def call(self,doc_id,pos_tag,neg_tag):

        doc_encode = tf.nn.embedding_lookup(self.text_embeddings, doc_id) 

        doc_encode = tf.squeeze(doc_encode) 

        #doc_encode = tf.math.l2_normalize(doc_encode,axis=1) 

        pos_tag_emb = tf.nn.embedding_lookup(self.tag_embeddings, pos_tag)  # batch_size*1*tag_dim

        pos_tag_emb = tf.squeeze(pos_tag_emb) 

        #pos_tag_emb = tf.math.l2_normalize(pos_tag_emb,axis=1)


        # inner production of document to positive tag
        pos_production = tf.reduce_sum(tf.multiply(doc_encode, pos_tag_emb), 1)


        # negative tag embeddings
        neg_tag_embs = tf.nn.embedding_lookup(self.tag_embeddings, neg_tag)
        #neg_tag_embs = tf.math.l2_normalize(neg_tag_embs,axis=2) 

        neg_tag_embs = tf.transpose(neg_tag_embs, (0, 2, 1))


        #neg_tag_embs = tf.squeeze(neg_tag_embs) 


        # inner production of document to negative tags
        production_to_neg_items = tf.reduce_sum(tf.multiply(tf.expand_dims(doc_encode, -1), neg_tag_embs), 1)


        #production_to_neg_items = tf.reduce_sum(tf.multiply(doc_encode, neg_tag_embs), 1)

        # best negative item
        max_neg_production = tf.reduce_max(production_to_neg_items, 1)


        pos_neg_prd_embedding = tf.reduce_sum(tf.multiply(tf.expand_dims(pos_tag_emb, -1), neg_tag_embs), 1)
        #pos_neg_prd_embedding = tf.reduce_sum(tf.multiply(pos_tag_emb, neg_tag_embs), 1)

        # best negative item
        pos_max_neg_production = tf.reduce_max(pos_neg_prd_embedding, 1)

        #bias = tf.nn.embedding_lookup(self.B,doc_id) 

        #pbias= tf.nn.embedding_lookup(self.B1,pos_tag)

        pred_distance =  pos_production
        pred_distance_neg =  max_neg_production
        pred_distance_PN =  pos_max_neg_production


        #a = tf.maximum(-pred_distance + pred_distance_neg + bias, 0)
        #b = tf.maximum(-pred_distance + pred_distance_PN + pbias,0)

        a = -tf.math.log(tf.sigmoid(pred_distance-pred_distance_neg))
        b = -tf.math.log(tf.sigmoid(pred_distance-pred_distance_PN))
         
        #whole model
        loss= self.lamba*tf.reduce_mean(a) + self.beta*tf.reduce_mean(b)
        #loss=loss-1*(10*(tf.reduce_mean(bias) +tf.reduce_mean(pbias)))

        return loss
    	

    def init_text(self,doc_v):

        self.text_embeddings.assign(doc_v)
        #self.text_embeddings = tf.math.l2_normalize(self.text_embeddings,axis=1)

    def get_text_emb(self):

        return self.text_embeddings

    def get_tag_emb(self):

        return self.tag_embeddings

    def org_predict(self,doc):


        tag_scores = tf.matmul(doc,tf.transpose(self.tag_embeddings,[1,0]))

        return tag_scores
class Train():

	def __init__(self,sampler,Text_encoder,doc_label_metric,max_iter,n_tags,X_train,X_test, Y_train,Y_test):
		super(Train, self).__init__()

		self.text_encoder=Text_encoder
		self.l_metric = doc_label_metric
		self.sampler=sampler
		self.max_iter=max_iter
		self.n_tags=n_tags
		self.X_train=X_train
		self.X_test=X_test
		self.Y_test=Y_test
		self.Y_train=Y_train
		self.optimizer=tf.keras.optimizers.Adam(0.001)
		self.optimizer_1=tf.keras.optimizers.Adam(0.01)
		self.optswa = tfa.optimizers.SWA(tf.keras.optimizers.Adam(0.01),1,2)
		self.optswa_1 = tfa.optimizers.SWA(tf.keras.optimizers.Adam(0.01),1,2)


	def loss_func(self, y_pred, y_true):

		loss_per_pair = tf.nn.sigmoid_cross_entropy_with_logits(tf.cast(y_true, tf.float32),y_pred)
		loss = tf.reduce_mean(tf.reduce_sum(loss_per_pair,axis = 1))

		return loss

	def train_step(self, doc_id, pos_tag, neg_tag):

		with tf.GradientTape() as tape:
			loss = self.l_metric(doc_id, pos_tag, neg_tag)


		variables=self.l_metric.trainable_variables
		gradients = tape.gradient(loss, variables)
		#grads = [tf.clip_by_norm(g, 1.0) for g in gradients]

		self.optimizer_1.apply_gradients(zip(gradients, variables))

		return loss


	def train_step_2(self, doc, doc_v):

		with tf.GradientTape() as tape:
			output = self.text_encoder(doc)
			loss =  tf.losses.mean_squared_error(doc_v,output)
			loss =  tf.reduce_mean(loss)


		variables=self.text_encoder.trainable_variables
		gradients = tape.gradient(loss, variables)
		#grads = [tf.clip_by_norm(g, 1.0) for g in gradients]

		self.optimizer.apply_gradients(zip(gradients, variables))

		return loss

	def train_loop(self,TOP_N,log_path):
		best_predict = np.full((4,len(TOP_N)), 0.0)
		log = open(log_path, 'a')
		s = []
		test_ds = tf.data.Dataset.from_tensor_slices(self.X_train).batch(256)
		for t_i in tqdm(test_ds):
			doc_encode = self.text_encoder(t_i,False)
			#s.extend(tf.math.l2_normalize(doc_encode,axis=1))
			s.extend(doc_encode)
		s= np.array(s)

		self.l_metric.init_text(s)

		for iteration in range(self.max_iter):

			for m_it in range(1):
				
				print ("%d iteration ..." % (iteration+1))
				log.write("%d iteration \n" % (iteration+1))

				print("Doc Label metric...")

				tic = time.time()
				total_loss = 0
				self.sampler.init_norm_sampler_processor()
				
				bar_count = self.sampler.get_norm_count()
				bar = tqdm(total=bar_count)
				for _ in range(bar_count):
					batch_doc_id, batch_pos_tag, batch_neg_tags = self.sampler.next_batch()
					batch_loss=self.train_step(batch_doc_id, batch_pos_tag, batch_neg_tags)
					total_loss+=batch_loss
					bar.update(1)
				self.sampler.close()
				toc = time.time()
				elapsed = toc - tic
				bar.close()
				print("Loss: %.5f Elpased: %.4fs \n" % (total_loss, elapsed))

				
				
				total_loss = 0
				
				self.sampler.init_reverse_sampler_processor()
				tic = time.time()
				bar_count = self.sampler.get_reverse_count()
				bar = tqdm(total=bar_count)
				for _ in range(bar_count):
					batch_doc_id, batch_pos_tag, batch_neg_tags = self.sampler.next_batch()
					batch_loss=self.train_step(batch_doc_id, batch_pos_tag, batch_neg_tags)
					total_loss+=batch_loss
					bar.update(1)
				self.sampler.close()
				toc = time.time()
				elapsed = toc - tic
				bar.close()
				print("Loss: %.5f Elpased: %.4fs \n" % (total_loss, elapsed))
				
				
				

			
			doc_v = self.l_metric.get_text_emb()

			for g_it in range(1):
				total_loss = 0

				tic = time.time()
				
				print("Train GruModel...")
				train_ds = tf.data.Dataset.from_tensor_slices((self.X_train,doc_v)).shuffle(len(self.X_train)).batch(512)
				for t_x, t_l in tqdm(train_ds):
					batch_loss = self.train_step_2(t_x, t_l)
					total_loss += batch_loss


				toc = time.time()
				elapsed = toc - tic
				print("Loss: %.5f Elpased: %.4fs \n" % (total_loss, elapsed))
				
				tic = time.time()
				_precision, _recall, _ndcg, _hd= self.evaluate(TOP_N)
				if _recall[1] > best_predict[1][1]:
					best_predict = [_precision, _recall, _ndcg, _hd]
				toc = time.time()
				elapsed = toc - tic
				print("Loss: %.5f Elpased: %.4fs \n" % (total_loss, elapsed))

				print("Top-%s, Precision: %s Recall: %s NDCG: %s  HD: %s\n" % (
				TOP_N, val_format(_precision, ".5"), val_format(_recall, ".5"), val_format(_ndcg, ".5"), val_format(_hd, ".5")))
				log.write("Top-%s, Precision: %s Recall: %s NDCG: %s HD: %s\n\n" % (TOP_N, val_format(_precision, ".5"), val_format(_recall, ".5"), val_format(_ndcg, ".5"), val_format(_hd, ".5")))
				log.flush()
		log.close()
		
		return best_predict

	def evaluate(self,TOP_N):
		_precision = [0] * len(TOP_N)
		_recall = [0] * len(TOP_N)
		_hd = [0.0] * len(TOP_N)
		_ndcg = [0] * len(TOP_N)
		count = 0
		tic = time.time()
		s = []
		test_ds = tf.data.Dataset.from_tensor_slices(self.X_test).batch(512)
		for t_i in tqdm(test_ds):
			doc_encode = self.text_encoder(t_i,False)
			tag_score = self.l_metric.org_predict(doc_encode)
			sc,index = tf.math.top_k(tag_score,10)
			s.extend(index)
		toc = time.time()
		elapsed = toc - tic
		total_loss=0.00
		print("Loss: %.2f Elpased: %.4fs \n" % (total_loss, elapsed))
		pre = np.array(s)

		for n in tqdm(range(len(TOP_N)),desc='P,R,N testing...'):
			for i in range(len(self.Y_test)):
				test_decision = pre[i,:TOP_N[n]]
				p_at_k, r_at_k = PrecisionRecall(self.Y_test[i], test_decision)
				# sum the metrics of decision
				_precision[n] += p_at_k
				_recall[n] += r_at_k
				_ndcg[n] += NDCG(self.Y_test[i], test_decision)

		count = len(self.Y_test)
		# calculate the final scores of metrics
		for n in range(len(TOP_N)):
			_precision[n] /= count
			_recall[n] /= count
			_ndcg[n] /= count

		return _precision, _recall, _ndcg, _hd