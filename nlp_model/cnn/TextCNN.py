import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()


#text-cnn parameter
embedding_size = 100
sequence_length = 3
num_classes = 2
filter_sizes = [2,3]
num_filter = 8

sentences = ["i love you","he loves me", "she likes baseball", "i hate you","sorry for that", "this is awful"]
labels = [1,1,1,0,0,0] # 1 is good 0 is not good

word_list = ' '.join(sentences).split()
word_list = list(set(word_list))
word_dict = {w:i for i,w in enumerate(word_list)}
vocab_size = len(word_dict)

inputs = []
for sen in sentences:
	inputs.append(np.asarray([word_dict[n] for n in sen.split()]))


outputs = []
for out in labels:
	outputs.append(np.eye(num_classes)[out])# one hot 


#Model
X = tf.placeholder(tf.int32,[None,sequence_length])
Y = tf.placeholder(tf.int32,[None,num_classes])
#随机生成词向量
W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name='embedding_matrix')
embedded_chars = tf.nn.embedding_lookup(W,X)#[batch_size,sequence_length,embedding_size]


pooled_outpus = []
for i,filter_size in enumerate(filter_sizes):
	with tf.variable_scope("CNN-max-pooling-%s" % filter_size):
		conv =  tf.layers.conv1d(embedded_chars,
							num_filter,
							filter_size,
							name='conv'
							)
		pool = tf.reduce_max(conv,reduction_indices=[1],name='gmp')
		pooled_outpus.append(pool) #dim of pooled:[batch_size(=6),output_height(=1),output_width(=1),channel]

h_pool_flat = tf.concat(pooled_outpus,-1)
num_filters_total = num_filter*len(filter_sizes)

#Model-Training
Weight = tf.get_variable("W",shape=[num_filters_total,num_classes])
Bias = tf.Variable(tf.constant(0.1,shape=[num_classes]))
model = tf.nn.xw_plus_b(h_pool_flat,Weight,Bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#Model-predict
hypothesis = tf.nn.softmax(model)
predictions = tf.argmax(hypothesis,1)

#training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(500):
	_, loss = sess.run([optimizer,cost],feed_dict={X:inputs,Y:outputs})
	if(epoch+1)%100==0:
		print("Epoch:","%06d" %(epoch+1), 'cost=,' '{:.6f}'.format(loss))


#test
test_text = 'sorry hate you'
tests = []
tests.append(np.asarray([ word_dict[n] for n in test_text.split()]))

predict = sess.run([predictions],feed_dict={X:tests})
result = predict[0][0]
if(result==0):
	print(test_text, "is Bad mean")
else:
	print(test_text,'is good mean')



















