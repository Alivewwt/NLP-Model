import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    文本分类模型
    词嵌入层，卷积层，池化层，分类层
    """
    def __init__(self,sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],"input_x")
        self.input_y = tf.placeholder(tf.int32,[None,num_classes],"input_y")
        self.dropout_rate = tf.placeholder(tf.float32,name="dropuut_rate")

        l2_loss = tf.constant(0.0)
        #随机初始化向量
        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name="embeddings")
            self.embeddings = tf.nn.embedding_lookup(self.W,self.input_x)
        '''
        #使用预训练的词向量
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.get_variable(
                shape = [vocab_size, embedding_size],
                initializer = tf.constant_initializer(embeddings),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        '''
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.variable_scope("cnn-max-pooling-%s" %filter_size):
                conv = tf.layers.conv1d(self.embeddings,num_filters,filter_size,name="conv")
                gmp = tf.reduce_max(conv,reduction_indices=[1],name="global_max_pooling")
                pooled_outputs.append(gmp)
        self.h_pool = tf.concat(pooled_outputs,-1)


        with tf.variable_scope("score"):
            #全连接层
            fc = tf.layers.dense(self.h_pool,256,name="fc1")
            fc = tf.layers.dropout(fc,self.dropout_rate)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc,num_classes,name="fc2")
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits),1) #预测类别

        with tf.variable_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.logits)
            self.loss = tf.reduce_mean(loss,)
            acc = tf.equal(tf.argmax(self.input_y,1),self.y_pred)
            self.accuracy = tf.reduce_mean(tf.cast(acc,tf.float32),name="accuracy")


if __name__ == '__main__':
    #句子长度，类别，字典数，embedding_dim,
    textCNN = TextCNN(5,2,6,10,[2,3,4],16)
    input_x = np.array([[1,2,3,2,4],
                       [3,4,1,5,0],
                        [4,5,1,1,0]])
    input_y = np.array([[1,0],
                        [0,1],
                        [0,1]])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss,acc = sess.run([textCNN.loss,textCNN.accuracy],feed_dict={textCNN.input_x:input_x,
                                                                       textCNN.input_y:input_y})
        print(loss)


