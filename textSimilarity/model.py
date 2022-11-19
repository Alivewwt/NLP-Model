#encoding=utf-8
import tensorflow as tf
import tensorflow.contrib.layers as layers
from bert import  modeling


class simModel(object):
	def __init__(self,max_seq_len,num_classes):

		self.max_seq_len = max_seq_len
		self.num_classes = num_classes

		self.ab_input_ids = tf.placeholder(tf.int32, [None, None], name="ab_input_ids")
		self.ab_input_masks = tf.placeholder(tf.int32, [None, None], name="ab_input_masks")
		self.ab_segment_ids = tf.placeholder(tf.int32, [None, None], name="ab_segment_ids")
		self.ac_input_ids = tf.placeholder(tf.int32, [None, None], name="ac_input_ids")
		self.ac_input_masks = tf.placeholder(tf.int32, [None, None], name="ac_input_masks")
		self.ac_segment_ids = tf.placeholder(tf.int32, [None, None], name="ac_segment_ids")
		self.label = tf.placeholder(tf.int32,[None,self.num_classes],name='labels')

		self.dropout_keep = tf.placeholder(tf.float32,name="dropout_keep_prob")
		self.golbal_step = tf.Variable(0,trainable=False,name="global_step")

		with tf.variable_scope("bert_output") as scope:
			ab = self.bert_layer(self.ab_input_ids,self.ab_input_masks,self.ab_segment_ids)
			scope.reuse_variables()
			ac = self.bert_layer(self.ac_input_ids,self.ac_input_masks,self.ac_segment_ids)

		self.ab_output = ab[:,0,:]
		self.ac_output = ac[:,0,:]

		self.subtract_output = self.ab_output-self.ac_output
		self.output = tf.nn.dropout(self.subtract_output,self.dropout_keep,name="dropout")

		self.loss,self.acc = self.loss_layer(self.output)

		self.opt = tf.train.AdadeltaOptimizer(1e-3).minimize(self.loss,global_step=self.golbal_step)
		# opt = tf.train.AdadeltaOptimizer(1e-3)
		# grads_and_vars = opt.compute_gradients(self.loss)
		# self.train_op = opt.apply_gradients(grads_and_vars,global_step=self.golbal_step)



	def bert_layer(self,input_ids,mask_ids,segment_ids):
		bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
		model = modeling.BertModel(
			config=bert_config,
			is_training=True,
			input_ids=input_ids,
			input_mask=mask_ids,
			token_type_ids=segment_ids,
			use_one_hot_embeddings=False)#是否使用GPU

		bert_embeddings = model.get_sequence_output()
		return bert_embeddings


	def loss_layer(self,output):
		with tf.name_scope("loss"):
			loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.cls,logits=output)
			losses = tf.reduce_mean(loss)

		with tf.name_scope("accuracy"):
			acc = tf.equal(tf.argmax(tf.nn.softmax(output),1),tf.argmax(self.cls,1) )
			accuracy = tf.reduce_mean(tf.cast(acc,tf.float32),name='accuracy')

		return losses,accuracy



