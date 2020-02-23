#encoding = utf-8
import tensorflow as tf
import tensorflow.contrib.layers as layers


class bilstm(object):
	def __init__(self,vocab_size,embedding_size,seq_length,num_classes,hidden_units,train):
		self.vocab_size =vocab_size
		self.embedding_size = embedding_size
		self.num_classes =num_classes
		self.hidden_units = hidden_units
		self.initializer = layers.xavier_initializer()

		#定义输入

		self.x = tf.placeholder(dtype=tf.int32,shape=[None,seq_length],name='inputs')
		self.y = tf.placeholder(dtype=tf.float32,shape=[None,self.num_classes],name='labels')
		self.dropout_prob = tf.placeholder(tf.float32,name='dropout_prob')

		lengths = tf.sign(tf.abs(self.x))
		self.lengths = tf.cast(tf.reduce_sum(lengths,1),tf.int32)

		embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size,elf.embedding_size],-1.0,1.0),name="embedding_W")
		self.embedding_input = tf.nn.embedding_lookup(embedding_matrix,self.x)

		lstm_output = self.bi_lstm(self.embedding_input,self.lengths)

		#if train:
		lstm_output = tf.nn.dropout(lstm_output,1-self.dropout_prob)

		logits = self.project_layer(lstm_output,128)
		self.loss,self.acc = self.loss_layer(logits)

		#opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

		tvars= tf.trainable_variables()
		for tv in tvars:
			print("name=%s, shape =%s" %(tv.name,tv.shape))

	def bi_lstm(self,lstm_input,lengths):
		with tf.variable_scope("bilstm"):
			lstm_cell ={}
			for direction in ['forward','backward']:
				with tf.variable_scope(direction):
					lstm_cell[direction] = tf.contrib.rnn.LSTMCell(
									self.hidden_units,
									use_peepholes=True,
									initializer=self.initializer,
									state_is_tuple=True)

			output,_ = tf.nn.bidirectional_dynamic_rnn(
							lstm_cell['forward'],
							lstm_cell['backward'],
							lstm_input,
							sequence_length=lengths,
							dtype=tf.float32)
		return tf.concat(output,-1)

	def project_layer(self,lstm_output,output_size):
		dense_input = tf.reduce_max(lstm_output,reduction_indices=[1])
		with tf.variable_scope("project_layer"):
			w = tf.get_variable(shape=[self.hidden_units*2,output_size],
								initializer = self.initializer,
								name="dense_w",
								dtype=tf.float32)
			b = tf.get_variable(shape=[output_size],
								initializer=tf.zeros_initializer(),
								name='b',
								dtype=tf.float32)
			#dense_output = tf.matmul(dense_input,w)+b
			outputs = tf.nn.tanh(tf.nn.xw_plus_b(dense_input,w,b))
			outputs = tf.layers.dense(outputs,self.num_classes)
		return outputs

	def loss_layer(self,logits):
		with tf.variable_scope("loss"):
			loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,labels = self.y)
			losses = tf.reduce_mean(loss)

		with tf.variable_scope('accuracy'):
			predicts = tf.argmax(tf.nn.softmax(logits),-1)
			acc = tf.equal(tf.argmax(self.y,-1),predicts)
			accuracy = tf.reduce_mean(tf.cast(acc,tf.float32))

		return losses,accuracy






	