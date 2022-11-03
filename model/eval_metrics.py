'''
@author: Hao Wu, Zhengxin Zhang, ShaoWei Qin

'''

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import norm


class  ID2ID():
    '''
    Mapping IDs to new unique identifiers
    '''

    def __init__(self):
        self.map = {}
        
    def assignID(self, key):
        if key not in self.map:
            self.map[key] = len(self.map)
        return self.map[key]
    
    def getID(self, key):
        return self.map.get(key, -1)
    
    def items(self):
        return self.map.items()
    
    def size(self):
        return len(self.map)
    
def AP(ground_truth, test_decision):
    p_at_k = [0.0] * len(test_decision)
    C = 0
    for i in range(1, len(test_decision) + 1):
        rel = 0
        if test_decision[i - 1] in ground_truth:
            rel = 1
            C += 1
        p_at_k[i - 1] = rel * C / i
    if C==0:
        return 0.0        
    else:
        return np.sum(p_at_k) / C 

def NDCG(ground_truth, test_decision):
    dcg = 0
    C = 0
    for i in range(1, len(test_decision) + 1):
        rel = 0
        if test_decision[i - 1] in ground_truth:
            rel = 1
            C += 1        
        dcg += (np.power(2, rel) - 1) / np.log2(i + 1)
    if C == 0:
        return 0
    idcg = 0
    for i in range(1, C + 1):
        idcg += (1 / np.log2(i + 1))
    return dcg / idcg

def PrecisionRecall(ground_truth, test_decision):
    hit_set = list(set(ground_truth) & set(test_decision))
    precision = len(hit_set) / float(len(test_decision))
    recall = len(hit_set) / float(len(ground_truth))
    return precision, recall

def HD(ground_truth, test_decision,n):
    hit_set = list(set(ground_truth[:n])&set(test_decision[:n]))
    hd_precision=1-len(hit_set)/n
    return hd_precision

def val_format(value_list, str_format):
    _str_list = []
    for val in value_list:
        _str_list.append(format(val, str_format))
    return _str_list
    
