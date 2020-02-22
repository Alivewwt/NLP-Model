import tensorflow as tf
import numpy as np
class TextCNN(object):
    """
    文本分类模型
    词嵌入层，卷积层，池化层，分类层
    """
    def __init__(
      self, embeddings,sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 占位符，输入，输出，dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 词向量层 随机初始化
        '''
         with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        '''
        #使用预训练的词向量
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.get_variable(
                shape = [vocab_size, embedding_size],
                initializer = tf.constant_initializer(embeddings),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
       	pooled_outputs = []
    	for kernel_size in filter_sizes:
		with tf.variable_scope("CNN-max-pooling-%s" % kernel_size):
                	conv = tf.layers.conv1d(self.embedd_chars,num_filters,kernel_size,name='conv')
            		gmp = tf.reduce_max(conv,reduction_indices=[1],name='gmp')
            		pooled_outputs.append(gmp)
            
         self.h_pool = tf.concat(pooled_outputs,1)
        

       with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
