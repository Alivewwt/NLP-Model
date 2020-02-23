#encoding = utf-8
import tensorflow as tf

class Seq2Seq(object):
	def build_inputs(self,config):
		#定义输入内容和内容长度
		self.seq_inputs = tf.placeholder(tf.int32,shape=[config.batch_size,None],name="seq_input")
		self.seq_inputs_length = tf.placeholder(tf.int32,shape=[config.batch_size,],name="seq_input_length")
		#定义解码内容和长度
		self.seq_target = tf.placeholder(tf.int32,shape=[config.batch_size,None],name="seq_target")
		self.seq_target_length =  tf.placeholder(tf.int32,shape=[config.batch_size,],name="seq_target_length")

	def build_loss(self,logits):
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.seq_target,logits= logits)
		loss = tf.reduce_mean(loss)
		return loss

	def builf_opt(self,lr,loss):
		self.opt = tf.train.AdamOptimizer(lr).minimize(loss)

	def attn(self,hidden,encoder_outputs):
		#hidden B*D
		#encoder output B*S*D
		attn_weights = tf.matmul(encoder_outputs,tf.expand_dims(hidden,2))
		# B*S*1
		attn_weights = tf.nn.softmax(attn_weights,axis=1)
		context = tf.squeeze(tf.matmul(tf.transpose(encoder_outputs,[0,2,1]),attn_weights))
		#context B*D
		return  context

	def __init__(self,config,w2i_target,useTeacherForcing=True,useAttention=True,useBeamSearch=1):
		self.build_inputs(config)
		with tf.variable_scope("encoder"):
			encoder_embedding = tf.Variable(tf.random_uniform(config.source_vocab_size,config.embedding_size),
			                                dtype=tf.float32,name="encoder_embedding")
			encoder_input_embeddings = tf.nn.embedding_lookup(encoder_embedding,self.seq_inputs)
			((encoder_fw_outputs,encoder_bw_outputs),
			 (encoder_fw_final_state,encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw = tf.nn.rnn_cell.GRUCell(config.hidden_dim),
				cell_bw = tf.nn.rnn_cell.GRUCell(config.hidden_dim),
				inputs = encoder_input_embeddings,
				sequence_length = self.seq_inputs_length,
				dtype = tf.float32,
				time_major = False
			)
			encoder_state = tf.add(encoder_fw_final_state,encoder_bw_final_state)
			encoder_outputs = tf.add(encoder_fw_outputs,encoder_bw_outputs)

		with tf.variable_scope("decoder"):
			decoder_embedding = tf.Variable(tf.random_uniform(config.target_vocab_size,config.embedding_size),
			                                dtype=tf.float32,
			                                name="decoder_embedding")

			with tf.variable_scope("gru_cell"):
				decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
				decoder_inital_state = encoder_state # encoder last step state
			if useTeacherForcing and not useAttention:
				pass
			tokens_go = tf.ones([config.batch_size],dtype=tf.int32,name="tokens_GO")*w2i_target["_GO"]
			tokens_eos = tf.ones([config.batch_size], dtype=tf.int32, name="tokens_EOS") * w2i_target["_EOS"]
			token_eos_embedding = tf.nn.embedding_lookup(decoder_embedding,tokens_eos)
			token_go_embedding = tf.nn.embedding_lookup(decoder_embedding,tokens_go)

			W = tf.Variable(tf.random_uniform(config.hidden_dim,config.target_vocab_size),dtype=tf.float32,name="decoder_output_w")
			b = tf.Variable(tf.zeros([config.target_vocab_size]),dtype=tf.float32,name="decoder_output_b")
			"""
			def loop_fn(time, cell_output, cell_state, loop_state):
				
				loop_fn 是一个函数，这个函数在 rnn 的相邻时间步之间被调用。
							　　
					函数的总体调用过程为：
					1. 初始时刻，先调用一次loop_fn，获取第一个时间步的cell的输入，loop_fn 中进行读取初始时刻的输入。
					2. 进行cell自环　(output, cell_state) = cell(next_input, state)
					3. 在 t 时刻 RNN 计算结束时，cell 有一组输出 cell_output 和状态 cell_state，都是 tensor；
					4. 到 t+1 时刻开始进行计算之前，loop_fn 被调用，调用的形式为
					loop_fn( t, cell_output, cell_state, loop_state)，
					而被期待的输出为：(finished, next_input, initial_state, emit_output, loop_state)；
					5. RNN 采用 loop_fn 返回的 next_input 作为输入，initial_state 作为状态，计算得到新的输出。
					在每次执行（output， cell_state） =  cell(next_input, state)后，执行 loop_fn() 进行数据的准备和处理。

					参数:
					time: 第 time 个时间步之前的处理，起始为 0
					cell_output: 上一个时间步的输出
					cell_state: RNNCells 的长时记忆
					loop_state: 保存了上个时间步执行后是否已经结束，如果输出 alignments，还保存了存有 alignments 的 TensorArray

					返回:
					next_done:是否已经完成
					next_cell_input:下个时间步的输入
					next_cell_state:下个时间步的状态
					emit_output:实际的输出
					next_loop_state:保存一些循环信息
				
				return (next_done, next_cell_input, next_cell_state, emit_output, next_loop_state)
			"""

			def loop_fn(time,previous_output,previous_state,previous_loop_state):
				if previous_state is None: #time_step==0
					initial_elements_finished = (0>=self.seq_target_length) #all flase at the initial step
					#encoder 端结束位置状态作为解码的初始状态
					initial_state = decoder_inital_state # last time step
					initial_input = token_go_embedding
					if useAttention:
						initial_input = tf.concat([initial_input,self.attn(initial_state,encoder_outputs)],1)

					initial_output = None
					initial_loop_state = None
					#是否完成、下个时间步的输入，下个时间步的状态，实际的输出，保存的循环状态
					return (initial_elements_finished,initial_input,initial_state,initial_output,initial_loop_state)

				else:
					def get_next_input():
						if useTeacherForcing:
							#训练时刻直接用标准答案作为输出
							prediction = self.seq_target[:,time-1]
						else:
							output_logits = tf.add(tf.matmul(previous_output,W),b)
							prediction = tf.argmax(output_logits,axis=1)
						next_input = tf.nn.embedding_lookup(decoder_embedding,prediction)
						return next_input

					elements_finished = (time>=self.seq_target_length)
					finished = tf.reduce_all(elements_finished)
					# tf.cond(pred,true_fn = None,false_fn = None)
					#判断是否到结束位置
					input = tf.cond(finished, lambda: token_eos_embedding, get_next_input)
					if useAttention:
						input = tf.concat([input, self.attn(previous_state, encoder_outputs)], 1)
					state = previous_state
					output = previous_output
					loop_state = None #保存一些循环信息

					return (elements_finished, input, state, output, loop_state)
			#raw_rnn 只接受上一刻的输出作为下一时刻的输入
			decoder_outputs_ta, decoder_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
			decoder_outputs = decoder_outputs_ta.stack()
			decoder_outputs = tf.transpose(decoder_outputs, perm=[1, 0, 2])  # S*B*D -> B*S*D

			decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
			decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, config.hidden_dim))
			decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
			decoder_logits = tf.reshape(decoder_logits_flat,
			                            (decoder_batch_size, decoder_max_steps, config.target_vocab_size))

		self.out = tf.argmax(decoder_logits, 2)

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.seq_target,
			logits=decoder_logits,
		)
		sequence_mask = tf.sequence_mask(self.seq_target_length, dtype=tf.float32)
		loss = loss * sequence_mask
		self.loss = tf.reduce_mean(loss)
