from keras import backend as K
from GCN import *
import re,os
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09/')

sentence = 'which you step on to activate it'
de = nlp.dependency_parse(sentence)

print ('Dependency Parsing:', de)

dep_sentences = []
for i in range(10):
    dep_sentences.append(de)

#依存关系标签
_DEP_LABELS = ['ROOT', 'DOBJ','ADV', 'ADV-GAP', 'AMOD', 'APPO', 'BNF', 'CONJ', 'COORD', 'DEP',
               'DEP-GAP', 'DIR', 'DIR-GAP', 'DIR-OPRD', 'DIR-PRD', 'DTV', 'EXT',
               'EXT-GAP', 'EXTR', 'GAP-LGS', 'GAP-LOC', 'GAP-LOC-PRD', 'GAP-MNR',
               'GAP-NMOD', 'GAP-OBJ', 'GAP-OPRD', 'GAP-PMOD', 'GAP-PRD', 'GAP-PRP',
               'GAP-SBJ', 'GAP-TMP', 'GAP-VC', 'HMOD', 'HYPH', 'IM', 'LGS', 'LOC',
               'LOC-OPRD', 'LOC-PRD', 'LOC-TMP', 'MNR', 'MNR-PRD', 'MNR-TMP', 'NAME',
               'NMOD', 'NSUBJ','OBJ', 'OPRD', 'P', 'PMOD', 'POSTHON', 'PRD', 'PRD-PRP',
               'PRD-TMP', 'PRN', 'PRP', 'PRT', 'PUT', 'SBJ', 'SUB', 'SUFFIX',
                'TITLE', 'TMP', 'VC', 'VOC','MARK','ADVCL']

_DEP_LABELS_DICT = {label:ix for ix, label in enumerate(_DEP_LABELS)}

SEQ_LEN = len(sentence.split()) #7
BATCH_SIZE = len(dep_sentences) #10

adj_arc_in = np.zeros((BATCH_SIZE* SEQ_LEN, 2), dtype='int64')
adj_lab_in = np.zeros((BATCH_SIZE* SEQ_LEN), dtype='int64')

adj_arc_out = np.zeros((BATCH_SIZE * SEQ_LEN, 2), dtype='int64')
adj_lab_out = np.zeros((BATCH_SIZE * SEQ_LEN), dtype='int64')

#Initialize mask matrix
mask_in = np.zeros((BATCH_SIZE * SEQ_LEN), dtype='float32')
mask_out = np.zeros((BATCH_SIZE * SEQ_LEN), dtype='float32')

mask_loop = np.ones((BATCH_SIZE * SEQ_LEN, 1), dtype='float32')

#Get adjacency matrix for incoming and outgoing arcs 获得传入和传出的邻接矩阵
for idx_sentence, dep_sentence in enumerate(dep_sentences):
    for idx_arc, arc in enumerate(dep_sentence):
        if(arc[0] != 'ROOT') and arc[0].upper() in _DEP_LABELS:
            #get index of words in the sentence
            arc_1 = int(arc[1]) - 1
            arc_2 = int(arc[2]) - 1

            idx = (idx_arc) + idx_sentence * SEQ_LEN
            #Make adjacency matrix for incoming arcs
            adj_arc_in[idx] = np.array([idx_sentence, arc_2]) #入边
            adj_lab_in[idx] = np.array([_DEP_LABELS_DICT[arc[0].upper()]]) #依存关系标签
            
            #Setting mask to consider that index
            mask_in[idx] = 1

            #Make adjacency matrix for outgoing arcs
            adj_arc_out[idx] = np.array([idx_sentence, arc_1])  #出边
            adj_lab_out[idx] = np.array([_DEP_LABELS_DICT[arc[0].upper()]]) #依存关系标签
            
            #Setting mask to consider that index
            mask_out[idx] = 1

'''
adj_arc_in = np.random.random((BATCH_SIZE,SEQ_LEN, 2))
adj_lab_in = np.random.random((BATCH_SIZE,SEQ_LEN,))

adj_arc_out = np.random.random((BATCH_SIZE,SEQ_LEN, 2))
adj_lab_out = np.random.random((BATCH_SIZE,SEQ_LEN,))

#Initialize mask matrix
mask_in = np.random.random((BATCH_SIZE,SEQ_LEN,1))
mask_out = np.random.random((BATCH_SIZE,SEQ_LEN,1))

mask_loop = np.random.random((BATCH_SIZE,SEQ_LEN,1))
'''

adj_arc_in=K.variable(value=adj_arc_in,dtype='int32')
adj_lab_in=K.variable(value=adj_lab_in,dtype='int32')

adj_arc_out=K.variable(value=adj_arc_out,dtype='int32')
adj_lab_out=K.variable(value=adj_lab_out,dtype='int32')

mask_in=K.variable(value=mask_in,dtype='int32')
#mask_in=tf.reshape(mask_in,(BATCH_SIZE,SEQ_LEN,1))
mask_out=K.variable(value=mask_out,dtype='int32')
#mask_out=tf.reshape(mask_out,(BATCH_SIZE,SEQ_LEN,1))
mask_loop=K.variable(value=mask_loop,dtype='int32')

embedding=np.random.random((10,7,10))  #batch_size sen_length embedding_dim 
#embedding=K.placeholder(shape=(4,7,10),dtype='float32')#[sen_length,batch_size,embedding_dim]
embedding=K.variable(value=embedding,dtype='float32')
gcn=GraphConvLayer(10,5,67)# num_inputs(embedding_dim),num_units, num_labels
data=gcn([embedding,adj_arc_in, adj_arc_out,
                 adj_lab_in, adj_lab_out,
                 mask_in, mask_out,  mask_loop])
print(K.eval(data))
#print(tf.shape(data)) #10 7 5
print(data.shape[0],data.shape[1],data.shape[2])
nlp.close()
