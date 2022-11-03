

import os
import pickle
import random
import sys
import numpy as np
from operator import itemgetter
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from tqdm import tqdm

class Data_Factory():

    def load(self, path):
        D = pickle.load(open(path + "/document.all", "rb"))
        print ("Load preprocessed document data - %s" % (path + "/document.all"))
        return D

    def save(self, path, D):
        print ("Saving preprocessed document data - %s" % (path + "/document.all"))
        pickle.dump(D, open(path + "/document.all", "wb"))
        print ("Done!")

    def read_pretrained_word2vec(self, path, vocab, dim):
        print(f'Read pre_trained word vectors from {path}')
        if os.path.isfile(path):
            raw_word2vec = open(path, 'r', encoding='UTF-8')
        else:
            print ("Path (word2vec) is wrong!")
            sys.exit()

        word2vec_dic = {}
        all_line = raw_word2vec.read().splitlines()
        mean = np.zeros(dim)
        count = 0
        i=0
        for line in tqdm(all_line):
            line = line.strip()
            tmp = line.split()
            if len(tmp) == 0 or len(tmp[len(tmp)-300:len(tmp)])!=dim:
                continue

            _word = tmp[0]
            _vec = np.array(tmp[len(tmp)-300:len(tmp)], dtype=float)
            if _vec.shape[0] != dim:
                print ("Mismatch the dimension of pre-trained word vector with word embedding dimension!")
                sys.exit()
            word2vec_dic[_word] = _vec
            mean = mean + _vec
            count = count + 1
        mean = mean / count
        W = np.zeros((len(vocab) + 1, dim))
        count = 0
        for _word, i in vocab:
            if _word in word2vec_dic:
                W[i + 1] = word2vec_dic[_word]
                count = count + 1
            else:
                W[i + 1] = np.random.normal(mean, 0.1, size=dim)

        print ("%d of [%d] words exist in the given pretrained model" % (count, len(vocab)))
        return W
    
    def preprocess(self, train_text, train_tag, test_text, test_tag, _max_df=0.5, _vocab_size=50000):
        '''
        Preprocess rating and document data.
        Input:
            - path_itemtext: path for textual data of items(data format - item_id::sentence,sentence....)
            - path_itemtag: path for tagging data of items(data format - item_id::tag,tag,tag....)
            - _max_df: terms will be ignored that have a document frequency higher than the given threshold (default = 0.5)
            - vocab_size: vocabulary size (default = 8000)
        Output:
            - D['X']:list of sequence of word index of each item ([[1,2,3,4,..],[2,3,4,...],...])
            - D['Y']: list of sequence of tag index of each item ([[1,2,3,4,..],[2,3,4,...],...])
            - D['X_vocab'],D['Y_tag']: list of tuple (word|tag, index) in the given corpus
        '''
        Y=[]
        no_co = []
        i=0
        Y_tag = set()

        raw_content = open(train_tag, 'r',encoding='utf-8')
        for line in raw_content:
            line = line.strip()

            tmp = line.split(' ')

            Y_tag.update(tmp)
            Y.append(np.array([int(t) for t in tmp]))

            i+=1
        raw_content.close()

        raw_content = open(test_tag, 'r',encoding='utf-8')
        for line in raw_content:
            line = line.strip()
            tmp = line.split(' ')
            Y_tag.update(tmp)
            Y.append(np.array([int(t) for t in tmp]))
            i+=1

        raw_content.close()

        raw_content = open(train_text, 'r',encoding='utf-8')

        map_item2txt = {}  # temporal item_id to text
        text_count=0
        #for line in raw_content:
        for line in raw_content.readlines():
            line = line.strip()

            map_item2txt[text_count] = line
            text_count+=1

        raw_content.close()
        print(text_count)
        train_num = text_count


        print ("\tRemoving stop words...")
        print ("\tFiltering words by TF-IDF score with max_df: %.1f, vocab_size: %d" % (_max_df, _vocab_size))

        # Make vocabulary by document
        corpus = [value for value in map_item2txt.values()]
        vectorizer1 = TfidfVectorizer(max_df=_max_df, stop_words='english', max_features=_vocab_size)
        #vectorizer1 = CountVectorizer(stop_words='english', max_features=_vocab_size)

        tfidf_matrix = vectorizer1.fit_transform(corpus)
        X_vocab = vectorizer1.vocabulary_

        idf_weight=vectorizer1.idf_


        raw_content = open(test_text, 'r',encoding='utf-8')
        for line in raw_content:
            line = line.strip()
            map_item2txt[text_count] = line
            text_count+=1
        raw_content.close()
        print(text_count)

        # Make train/test data for run
        X  = []
        x_org= []
        np.random.seed(2021)
        index = 0
        train_wv_index = []

        for item in map_item2txt.keys():
            wordid_list = [X_vocab[word] + 1 for word in map_item2txt[item].split() if word in X_vocab]
            wordid = [word for word in map_item2txt[item].split() if word in X_vocab]
            x_org.append(wordid)
            #if len(wordid_list)>1000:
                #continue
            X.append(wordid_list)
            train_wv_index.append(index)

            index = index + 1


        X_vocab = sorted(X_vocab.items(), key=itemgetter(1))



        D = { 'X_vocab': X_vocab,'Y_tag': Y_tag,
              'X':np.array(X), 'Y':np.array(Y), 'train_num':train_num
            }
        
        print ("Done!")
        return D
    
    def pad_sequence(self, X_train, X_test, maxlen_doc):
        '''
        threshold: 0.95 means that maxlen_doc we taken covers 95% percentage of documents
        '''
        len_doc = [len(profile) for profile in X_train]
        len_doc.extend([len(profile) for profile in X_test])
        len_doc = sorted(len_doc)

        print ("X_train, X_test, maxlen_doc:%d " % (maxlen_doc))
        X_train = pad_sequences(X_train, maxlen=maxlen_doc,padding='pre',truncating='post')
        X_test = pad_sequences(X_test, maxlen=maxlen_doc,padding='pre',truncating='post')
        return X_train, X_test, maxlen_doc
   
    def get_X_mask(self,X_train,X_test,maxlen_doc):
        len_doc_train = [len(profile) for profile in X_train]
        len_doc_test = [len(profile) for profile in X_test]
        x_train_mask = np.zeros((len(X_train), maxlen_doc)).astype('float32')
        x_test_mask = np.zeros((len(X_test), maxlen_doc)).astype('float32')
        for i in range(len(X_train)):
            x_train_mask[i][:len_doc_train[i]]=1
        for i in range(len(X_test)):
            x_test_mask[i][:len_doc_test[i]]=1

        return x_train_mask,x_test_mask
    
    def generate_tag_item_matrix(self, train_items, tags):
        n_items = len(train_items)
        n_tags = len(tags)
        A = dok_matrix((n_tags, n_items), dtype=np.float32)
                 
        for item_id in range(n_items):
            for tag_id in train_items[item_id]:
                A[tag_id, item_id] = 1.0
        return A
    
    def get_binary_label(self,Y_train,n_tag):

        y_train = np.zeros((len(Y_train),n_tag))
        
        for index in range(len(Y_train)):
            y_train[index][Y_train[index]] = 1

        return y_train

    def load_group(self,dataset,Y_train,n_tags):

        group_y = np.load(f'{dataset}/label_group.npy', allow_pickle=True)
        
        y_cluster_id = np.zeros(n_tags,dtype=np.int64)
        
        for idx, labels in enumerate(group_y):
            for label in labels:
                y_cluster_id[int(label)] = idx

        y_group = [[] for i in range(len(Y_train))]

        for i in range(len(Y_train)):
            for label in Y_train[i]:
                y_group[i].append(y_cluster_id[int(label)])

        return group_y,np.array(y_group),y_cluster_id



if __name__ == '__main__':
    path = f'../data/Amazon-670K'
    log_path = '../log/colabel_metric_670k.log'
    data_Factory = Data_Factory()
    D = data_Factory.preprocess(train_text=f'{path}/train_raw_texts.txt', train_tag=f'{path}/train_labels.txt',test_text=f'{path}/test_raw_texts.txt', test_tag= f'{path}/test_labels.txt')




     