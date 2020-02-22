#encoding =utf-8
import tensorflow as tf 
import numpy as np
from bert import modeling


class bertText(object):
	def __init__(self,num_classes,max_sen_len,hidden_size,train=True):
		self.num_classes = num_classes
		self.max_sen_len = max_sen_len
		self.hidden_size = hidden_size
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.global_step = tf.Variable(0,name='global_step',trainable=False)

		self.input_ids = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sen_len],name="input_ids")
		self.segment_ids = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sen_len],name="segment_ids")
		self.mask_ids = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sen_len],name="mask_ids")
		self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.num_classes],name='labels')

		bert_output = self.bert_embeddings()

		bert_output = tf.nn.dropout(bert_output,keep_prob=self.dropout_keep_prob)

		outputs = self.project_layers(self.hidden_size,bert_output)

		self.loss,self.accuracy = self.loss_layer(outputs)

		# bert模型参数初始化的地方
		init_checkpoint = "uncased_L-12_H-768_A-12/bert_model.ckpt"
		# 获取模型中所有的训练参数。
		tvars = tf.trainable_variables()
        # 加载BERT模型
		(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                      init_checkpoint)
		tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
		print("**** Trainable Variables ****")
        # 打印加载模型的参数
		train_vars = []
		for var in tvars:
			init_string = ""
			if var.name in initialized_variable_names:
				init_string = ", *INIT_FROM_CKPT*"
			else:			
				train_vars.append(var)
			print("  name = %s, shape = %s%s", var.name, var.shape,
					init_string)

		self.opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss,global_step=self.global_step)


	def bert_embeddings(self):
		bert_config = modeling.BertConfig.from_json_file("uncased_L-12_H-768_A-12/bert_config.json")
		model = modeling.BertModel(
						config =bert_config,
						is_training =True,
						input_ids =self.input_ids,
						input_mask =self.mask_ids,
						token_type_ids = self.segment_ids,
						use_one_hot_embeddings = False)
		bert_emebeddings = model.get_pooled_output() #获得句子表示
		return bert_emebeddings

	def project_layers(self,hidden_size,bert_inputs):
		units = bert_inputs.shape[-1].value
		with tf.variable_scope("projects"):
			w = tf.get_variable(shape=[units,hidden_size],
								initializer=tf.contrib.layers.xavier_initializer(),
								name='w')
			b = tf.get_variable(shape=[hidden_size],
								initializer = tf.zeros_initializer(),
								name='b')
			output = tf.nn.xw_plus_b(bert_inputs,w,b)
			output = tf.nn.tanh(output)
			output = tf.layers.dense(bert_inputs,hidden_size,activation =tf.nn.relu,name="scores")
			project_out = tf.layers.dense(output,self.num_classes,name="cls")

			return project_out

	def loss_layer(self,outputs):
		with tf.name_scope("loss"):
			loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,logits=outputs)
			losses = tf.reduce_mean(loss)
		with tf.name_scope("accuracy"):
			pred = tf.argmax(tf.nn.softmax(outputs),1)
			acc = tf.equal(pred,tf.argmax(self.labels,1))
			accuracy = tf.reduce_mean(tf.cast(acc,'float'),name='accuracy')

		return losses,accuracy

