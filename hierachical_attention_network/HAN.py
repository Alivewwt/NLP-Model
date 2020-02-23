#encoding = utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import numpy as np


def length(sequences):
	used = tf.sign(tf.reduce_max(tf.abs(sequences),axis=2))
	seq_len = tf.reduce_sum(used,axis=1)
	seq_len = tf.cast(seq_len,tf.int32)
	return seq_len

class HAN(object):
	def __init__(self,vocab_size,num_classes,embedding_size=200,hidden_size=50):

		self.vocab_size = vocab_size #词表的数量
		self.num_classes = num_classes # 类别数
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size

		with tf.name_scope("input_placehodler"):
			self.max_sentence_num = tf.placeholder(tf.int32,name="max_sentence_num")
			self.max_sentence_length = tf.placeholder(tf.int32,name="max_sength_length")
			self.batch_size = tf.placeholder(tf.int32,name="batch_size")

			# x的shape为[batch_size,sentence_num,sentence_length(单词个数)] 但是每个样本的数据都不一样，这里指定为none
			self.input_x = tf.placeholder(tf.int32,[None,None,None],name="inpiut_x")
			self.input_y = tf.placeholder(tf.int32,[None,self.num_classes],name="labels")

		#构建模型
		word_embedded = self.word2vec()
		self.sen_vec = self.sen2vec(word_embedded)
		self.doc_vec = self.doc2vec(self.sen_vec)

		out = self.classifier(self.doc_vec)
		self.out = out

	def word2vec(self):
		with tf.name_scope("embedding"):
			# W = tf.Variable(tf.truncated_normal((self.vocab_size,self.embedding_size)))
			W = tf.get_variable(
				shape = [self.vocab_size,self.embedding_size],
				name = "metrics",
				initializer= layers.xavier_initializer()
			)
			#shape 为[batch_size,max_sen_num,max_sen_len,embedding]
			word_embedd = tf.nn.embedding_lookup(W,self.input_x)

		return word_embedd

	def sen2vec(self,word_embedd):
		with tf.name_scope("sent2vec"):
			#GRU的输入tensor是[batch_size,max_time,embeddi_size]
			word_embedd = tf.reshape(word_embedd,[-1,self.max_sentence_length,self.embedding_size])
			word_encoded = self.BirdirectionalGRU(word_embedd,name="word_encoder")
			# shape为[batch_size*max_sen_num,hidden_size*2]
			sen_vec = self.AttentionLayer(word_encoded,name="word_attention")
			return sen_vec

	def doc2vec(self,sen_vec):
		with tf.name_scope("doc2vec"):
			sen_vec = tf.reshape(sen_vec,[-1,self.max_sentence_num,self.hidden_size*2])
			doc_encoded = self.BirdirectionalGRU(sen_vec,name="doc_encoder")
			# shape 为[batch_size,hidden_size*2]
			doc_vec = self.AttentionLayer(doc_encoded,name="sen_attention")
			return doc_vec

	def classifier(self,doc_vec):
		with tf.name_scope("doc_classifier"):
			out = layers.fully_connected(doc_vec,self.num_classes,activation_fn=None)
			return out

	def BirdirectionalGRU(self,inputs,name):
		with tf.variable_scope(name):
			GRU_cell_fw = rnn.GRUCell(self.hidden_size)
			GRU_cell_bw = rnn.GRUCell(self.hidden_size)
			output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = GRU_cell_bw,
						cell_bw = GRU_cell_bw,
						inputs= inputs,
						sequence_length=length(inputs),
						dtype=tf.float32
						)
			outputs = tf.concat(output,-1)
			return outputs


	def AttentionLayer(self,inputs,name):
		with tf.variable_scope(name):
			# u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
			# 因为使用双向GRU，所以其长度为2×hidden_szie
			u_context = tf.Variable(tf.truncated_normal([self.hidden_size*2]),name="u_context")
			h = layers.fully_connected(inputs,self.hidden_size*2,activation_fn=tf.nn.tanh)
			att = tf.multiply(h,u_context)
			# shape为[batch_size, max_time, 1]
			alpha = tf.nn.softmax(tf.reduce_sum(att,axis=2,keep_dims=True),dim=1)
			att_out = tf.reduce_sum(tf.multiply(inputs,alpha),axis=1)
			return att_out


if __name__ == '__main__':
	max_sen_num = 5
	max_sen_len = 20
	batch_size = 8
	input_x = np.random.randint(0, 20, (8, 5, 20))
	input_y = np.random.randint(0, 2, (8, 2))
	with tf.Session() as sess:
		han = HAN(20,2)
		sess.run(tf.global_variables_initializer())
		feed_dict ={han.max_sentence_num:max_sen_num,
		            han.max_sentence_length:max_sen_len,
		            han.input_x:input_x,
		            han.input_y:input_y}
		sen_vec,doc_vec = sess.run([han.sen_vec,han.doc_vec],feed_dict=feed_dict)

		print(sen_vec.shape) #(40,100)
		print(sen_vec)
		print(doc_vec.shape) #(8,100)
		print(doc_vec)


