#encoding = utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
#定义神经元
bw_cell = rnn_cell(8)
fw_cell = rnn_cell(8)
#定义输入
Input = tf.placeholder(shape=[None,None],dtype=tf.float32,name="input")
#随机初始化embedding
init_embeddings = tf.random_uniform([6,10],-1,1)
embeddings = tf.get_variable(initializer=init_embeddings,name="embedding")
#获取输入变长
lengths = tf.reduce_sum(tf.sign(tf.abs(Input)),1)
lengths = tf.cast(lengths,tf.int32)
#随机生成输入
input_sample = np.random.randint(1,6,size=[3,4]) #1,5

input_embedding = tf.nn.embedding_lookup(embeddings,input_sample)

output,state = tf.nn.bidirectional_dynamic_rnn(
				cell_fw = bw_cell,
				cell_bw = fw_cell,
				inputs = input_embedding,
				sequence_length = lengths,
				dtype=tf.float32
				)
num_layers = 2
mul_cell_bw = [rnn_cell(8) for _ in range(num_layers)]
mul_cell_fw = [rnn_cell(8) for _ in range(num_layers)]

(output_m,state_fw,state_bw)= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(mul_cell_bw,
                                                                   mul_cell_fw,
                                                                   input_embedding,
                                                                   sequence_length = lengths,
                                                                   dtype=tf.float32)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	(fw_output,bw_output),(bw_state,fw_state) = sess.run([output,state],feed_dict={Input:input_sample})
	print(fw_output.shape) # [3,4,8]
	print(bw_state.c.shape) #[3,8]
	print(bw_state.h.shape) #[3,8]

	(output, m_bw_state,m_fw_state)= sess.run([output_m,state_fw,state_bw],
	                                                     feed_dict={Input:input_sample})
	print(output.shape) #[3,4,16]
	outputs = tf.concat(output,-1)
	print(outputs.shape) #[3,4,16]

	print(m_bw_state[0].c.shape)#[3,8]
	print(m_bw_state[0].h.shape) #[3,8]
	encoder_state_c = tf.concat([m_fw_state[0].c,m_bw_state[0].c],1)
	encoder_state_h = tf.concat([m_fw_state[0].h,m_bw_state[0].h],1)

	#给lstm_cell的initial_state赋予我们想要的值，而不简单的用0来初始化。
	encoder_state = rnn.LSTMStateTuple(c =encoder_state_c,h =encoder_state_h)
	print(encoder_state.c.shape) #[3,16]