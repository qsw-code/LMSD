import numpy
from multiprocessing import Process, Queue, Manager
from scipy.sparse import dok_matrix,lil_matrix
import numpy as np
from tqdm import tqdm
import random

np.random.seed(2022)

def sample_function(count,user_item_pairs,user_to_positive_set,n_tags, batch_size, n_negative, result_queue):

    numpy.random.shuffle(user_item_pairs)
    for i in count:
        #user_positive_items_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
        if (i + 1) * batch_size < len(user_item_pairs):
            user_positive_items_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
        else:  # for the last mini_batch where the size is less than self.batch_size
            user_positive_items_pairs = user_item_pairs[i * batch_size:, :]
        # sample negative samples
        negative_samples = numpy.random.randint(0,n_tags,size=(len(user_positive_items_pairs), n_negative))
        # Check if we sample any positive items as negative samples.
        # Note: this step can be optional as the chance that we sample a positive item is fairly low given a
        # large item set.
        for user_positive, negatives, c in zip(user_positive_items_pairs,negative_samples,range(len(negative_samples))):
            user = user_positive[0]
            for j, neg in enumerate(negatives):
                while neg in user_to_positive_set[user]:
                    negative_samples[c, j] = neg = numpy.random.randint(0, n_tags)

        pos_tag = []
        doc_id = []
        for (ti, tj) in user_positive_items_pairs:

            doc_id.append([ti])
            pos_tag.append([tj])

        result_queue.put((np.array(doc_id),np.array(pos_tag),negative_samples))

class Para_Sampler(object):

    def __init__(self, Y_train,n_tags, batch_size, n_negative, n_workers):
        
        self.user_item_pairs,self.user_to_positive_set,self.tag_doc_matrix = self.get_norm_sampler_pairs(Y_train,n_tags)
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.n_tags = n_tags
        self.n_workers = n_workers
        self.Y_train = Y_train
        #numpy.random.shuffle(self.user_item_pairs)
        #self.reverse_user_item_pairs = self.get_reverse_sampler_pairs(self.tag_doc_matrix,self.Y_train,self.n_tags,len(self.user_item_pairs),self.batch_size)

    def init_norm_sampler_processor(self): 
        self.result_queue = Queue()
        count=[]

        
        cc = int(len(self.user_item_pairs) / self.batch_size)+1
        for i in range(self.n_workers):
           if i!= self.n_workers-1:
               j = (i+1)*(int(cc/self.n_workers))
               count.append(np.arange(i*int(cc/self.n_workers),j))
           else:
                count.append(np.arange(i*int(cc/self.n_workers),cc))
        

        self.processors = []
        for i in range(self.n_workers):
            self.processors.append(Process(target=sample_function, args=(count[i],self.user_item_pairs,self.user_to_positive_set,self.n_tags,self.batch_size,self.n_negative,self.result_queue)))
            self.processors[-1].start()

    def init_reverse_sampler_processor(self): 
        self.reverse_user_item_pairs = self.get_reverse_sampler_pairs(self.tag_doc_matrix,self.Y_train,self.n_tags,len(self.user_item_pairs),self.batch_size)
        self.result_queue = Queue()
        count=[]

        cc = int(len(self.reverse_user_item_pairs) / self.batch_size)+1
        for i in range(self.n_workers):
           if i!= self.n_workers-1:
               j = (i+1)*(int(cc/self.n_workers))
               count.append(np.arange(i*int(cc/self.n_workers),j))
           else:
                count.append(np.arange(i*int(cc/self.n_workers),cc))

        self.processors = []
        for i in range(self.n_workers):
            self.processors.append(Process(target=sample_function, args=(count[i],self.reverse_user_item_pairs,self.user_to_positive_set,self.n_tags,self.batch_size,self.n_negative,self.result_queue)))
            self.processors[-1].start()
    
    def get_norm_sampler_pairs(self,Y_train,n_tags):
        
        n_items = len(Y_train)

        item_tag_matrix = dok_matrix((n_items, n_tags), dtype=np.float32)
        tag_doc_matrix = dok_matrix((n_tags, n_items), dtype=np.float32)
        for i in range(n_items):
            for j in Y_train[i]:
                item_tag_matrix[i, j] = 1.0
                tag_doc_matrix[j, i] = 1.0
                
        item_tag_matrix = lil_matrix(item_tag_matrix)
        item_tag_pairs = np.asarray(item_tag_matrix.nonzero()).T
        item_to_positive_set = {i: set(row) for i, row in enumerate(item_tag_matrix.rows)}

        return item_tag_pairs,item_to_positive_set,tag_doc_matrix

    def get_reverse_sampler_pairs(self,tag_doc_matrix,Y_train,n_tags,count,batch_size):
        
        n_items = len(Y_train)
        tag_doc = [[] for _ in range(n_tags)]
        for i in range(n_items):
            for j in Y_train[i]:
                tag_doc[j].append(i)

        prob = self.get_reverse_prob(tag_doc)

        tag_ids=[]
        doc_ids=[]
        #np.random.seed(2021)
        for i in tqdm(range(count),desc = 'reverse sampling...'):
            tag_id = np.random.choice(np.arange(n_tags),batch_size,p=prob)
            for t in tag_id:
                doc_ids.extend(tag_doc[t])
                tag_ids.extend(np.repeat(t,len(tag_doc[t])))
            if len(tag_ids) > count:
                break

        tag_doc_pairs = np.asarray([np.array(doc_ids),np.array(tag_ids)]).T
        
        return tag_doc_pairs

    def get_reverse_prob(self,item_tag_matrix):

        count_num=[]
        for t in item_tag_matrix:
            count_num.append(len(t))

        count_num = np.array(count_num)
        max_num = np.max(count_num)

        mid = np.zeros(len(count_num))
        for i in range(len(count_num)):
            if count_num[i] != 0:
                mid[i] = max_num/count_num[i]
        cm = np.sum(mid)
        
        prob = mid/cm

        return prob


    def next_batch(self):
        return self.result_queue.get()

    def get_norm_count(self):
        return int(len(self.user_item_pairs) / self.batch_size)+1
   
    def get_reverse_count(self):
        return int(len(self.reverse_user_item_pairs) / self.batch_size)+1

    def close(self):
        print("closing")
        self.result_queue.close()
        for p in self.processors: 
            p.terminate()
            p.join()