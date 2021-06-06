# encoding =utf-8
import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers


class Model(object):
	def __init__(self, batch_size, lstm_hid_dim, d_a, n_classes, label_embed, embeddings):
		self.n_classes = n_classes
		# 随机初始化词向量和标签向量
		self.embed = self._load_embeddings(embeddings)
		self.label_embed = self._load_labelembedd(label_embed)
		self.batch_size = batch_size
		self.lstm_hid_dim = lstm_hid_dim
		self.d_a = d_a

	def creat_placeholds(self):
		self.input_x = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_x")
		self.input_y = tf.placeholder(shape=[None, None], dtype=tf.int32, name="labels")

	def build_farward(self):
		lstm_input = tf.nn.embedding_lookup(self.embed, self.input_x)
		# get seq length
		used = tf.sign((tf.abs(self.input_x)))
		lengths = tf.reduce_sum(used, reduction_indices=1)
		# get lstm outputs states #[b,s,hid_dim]
		self.lstm_outputs, self.lstm_states = self.bilstm_layer(lstm_input, self.lstm_hid_dim, lengths)
		#get first linear layer
		selfatt = tf.layers.dense(self.lstm_outputs,self.d_a,activation=tf.nn.tanh)
		#get second linear layer
		selfatt = tf.layers.dense(selfatt,self.n_classes) #[b,s,n_class]
		selfatt = tf.nn.softmax(selfatt,dim=1)

		selfatt = tf.transpose(selfatt,[0,2,1]) #[b,n_class,s]
		self_att = tf.matmul(selfatt,self.lstm_outputs)
		#get label attention
		h1 = self.lstm_outputs[:, :, :self.lstm_hid_dim]
		h2 = self.lstm_outputs[:, :, self.lstm_hid_dim:]

		label = tf.tile(tf.expand_dims(self.label_embed,0),[self.batch_size,1,1])
		# [b,n_class,dim] [b,dim,seq] -> [b,n_class,seq]
		m1 = tf.matmul(label,tf.transpose(h1,[0,2,1]))
		m2 = tf.matmul(label,tf.transpose(h2,[0,2,1]))
		# [b,n_class,seq] *[b,seq,hidden_dim]
		a = tf.matmul(m1,h1)
		b = tf.matmul(m2,h2)
		label_att = tf.concat([a,b],2)
		weight1 = tf.nn.sigmoid(tf.layers.dense(label_att,1))
		weight2 = tf.nn.sigmoid(tf.layers.dense(self_att,1))
		weight1 = weight1/(weight1+weight2)
		weight2 = 1-weight1

		doc = weight1*label_att+weight2*self_att
		avg_sentence_embed = tf.reduce_sum(doc,1)/self.n_classes
		self.pred = tf.nn.sigmoid(tf.layers.dense(avg_sentence_embed,self.n_classes))
		# return self.pred


	def _load_embeddings(self, embed):
		embeddings = tf.Variable(tf.random_uniform([embed[0], embed[1]], -1.0, 1.0), name="embeddings")
		return embeddings

	def _load_labelembedd(self, label_embed):
		label_embed = tf.Variable(tf.random_uniform([label_embed[0], label_embed[1]], -1.0, 1.0), name="label_embed")
		return label_embed

	def bilstm_layer(self, lstm_inputs, lstm_hid_dim, lengths):
		with tf.variable_scope("bilstm") as scope:
			lstm_cell = {}
			for dir in ["forward", "backward"]:
				with tf.variable_scope(dir):
					lstm_cell[dir] = rnn.LSTMCell(
						lstm_hid_dim,
						use_peepholes=True,
						state_is_tuple=True,
						initializer=layers.xavier_initializer())
			output, (encoder_fw_state, encoder_bw_state) = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
			                                                                               lstm_cell["backward"],
			                                                                               lstm_inputs,
			                                                                               sequence_length=lengths,
			                                                                               dtype=tf.float32)
			final_states = tf.concat([encoder_fw_state.h, encoder_bw_state.h], -1)
			outputs = tf.concat(output, -1)

		return outputs, final_states


if __name__ == '__main__':
	# 多标签文本分类
	inputs = np.array([[12, 13, 14, 16, 0],
	                   [22, 3, 33, 45, 67],
	                   [98, 78, 88, 98, 99]])
	labels = np.array([[1, 2, 3], [1, 4, 0], [2, 5, 6]])
	label_embed = [7, 256]
	embeddings = [100, 256]
	model = Model(batch_size=3, lstm_hid_dim=256, d_a=200, n_classes=7,
	              label_embed=label_embed, embeddings=embeddings)
	# label_embed = np.load("label_embed.npy")
	# print(label_embed)
	# print(label_embed.shape) #54*300
	with tf.Session() as sess:
		model.creat_placeholds()
		model.build_farward()
		sess.run(tf.global_variables_initializer())
		lstm_outputs, lstm_states,preds = sess.run([model.lstm_outputs, model.lstm_states,model.pred],
		                                     feed_dict={model.input_x: inputs, model.input_y: labels})
		print("lstm outputs:\n%s" % lstm_outputs)
		print("lstm_state:\n%s" % lstm_states)
		print("pred:\n%s"%preds)
		print(lstm_states.shape, lstm_outputs.shape,preds.shape)

