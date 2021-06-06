#! -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

import os
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
from sklearn.metrics import precision_recall_fscore_support
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

#加载数据
data = pkl.load(open('./data/data.bin','rb'))
print("data loaded!")

sentences = data["sen"]
labels = data['labels']
word_embeddings = data['wordEmbeddings']

sentences = np.array(sentences)
print("sentences shape:", sentences.shape)

max_sentence_len =sentences.shape[1]

labels = keras.utils.to_categorical(labels)
print('labels shape: ',labels.shape)


indices = np.arange(sentences.shape[0])
np.random.shuffle(indices)
sentences = sentences[indices]
labels = labels[indices]
train_len = int(len(sentences)*0.9)
x_train = sentences[:train_len]
y_train = labels[:train_len]
x_test = sentences[train_len:]
y_test = labels[train_len:]

y_test = y_test.argmax(axis=-1)

#  :: Create the network :: 
print('Build model...')

# set parameters:
batch_size = 50
nb_filter = 50
filter_lengths = [2,3,4]
hidden_dims = 100
nb_epoch = 20

words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')

#Our word embedding layer
wordsEmbeddingLayer = Embedding(word_embeddings.shape[0],
                    word_embeddings.shape[1],                                     
                    weights=[word_embeddings],
                    trainable=False)

words = wordsEmbeddingLayer(words_input)

#Now we add a variable number of convolutions
words_convolutions = []
for filter_length in filter_lengths:
    words_conv = Convolution1D(filters=nb_filter,
                            kernel_size=filter_length,
                            padding='same',
                            activation='relu',
                            strides=1)(words)
                            
    words_conv = GlobalMaxPooling1D()(words_conv)      
    
    words_convolutions.append(words_conv)  

output = concatenate(words_convolutions)

# We add a vanilla hidden layer together with dropout layers:
output = Dropout(0.5)(output)
output = Dense(hidden_dims, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(output)
output = Dropout(0.25)(output)


# We project onto a single unit output layer, and squash it with a sigmoid:
output = Dense(2, activation='sigmoid',  kernel_regularizer=keras.regularizers.l2(0.01))(output)

model = Model(inputs=[words_input], outputs=[output])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

for epoch in range(nb_epoch):
    print("\n------------- Epoch %d ------------" % (epoch+1))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
    pred = model.predict([x_test],verbose = False)
    pred_test = pred.argmax(axis=-1)
    
    prec,recall,f1, _ = precision_recall_fscore_support(
                    y_test,
                    pred_test,
                    labels=list(range(1,2)),
                    average='micro'
                    )
    print('precision: %.3f recall: %.3f f1: %.3f' %(prec,recall,f1))
