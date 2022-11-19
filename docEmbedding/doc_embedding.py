#encoding =utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn


class Model(object):
     def __init__(self,config):
         print(config)
         self.config = config
         self.lr = config["lr"]
         self.char_dim = config["char_dim"]

         self.lstm_dim = config["lstm_dim"]
         self.seg_dim = config["seg_dim"]
         self.subtype_dim = config["subtype_dim"]
         self.num_tags = config["num_tags"]
         self.num_chars = config["num_char"]
         self.num_steps = config["num_steps"]
         self.num_segs = 14
         self.num_subtypes = 51
         self.seq_nums = 8
         self.multi_hop = 4

         self.global_step = tf.Variable(0,trainable=False)
         self.best_dev_f1 =  tf.Variable(0.0,trainable=False)
         self.best_test_f1 = tf.Variable(0.0,trainable=False)
         self.initiaizer = layers.xavier_initializer()

         self.char_inputs = tf.placeholder(dtype=tf.int32,
                                           shape=[None, None],
                                           name="ChatInputs")
         self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="SegInputs")
         self.subtype_inputs = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name="SubInputs")
         self.targets = tf.placeholder(dtype=tf.int32,
                                       shape=[None, None],
                                       name="Targets")

         self.doc_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None, self.num_steps],
                                          name="doc_inputs")
         self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

         self.char_lookup = tf.get_variable(
             name="char_embedding",
             shape=[self.num_chars, self.char_dim],
             initializer=self.initiaizer)


         used =tf.sign(tf.abs(self.char_inputs))
         length = tf.reduce_sum(used,reduction_indices=1)
         self.length = tf.cast(length,tf.int32)
         self.batch_size = tf.shape(self.char_inputs)[0]

         self.embedding = self.embedding_layer(self.char_inputs, self.seg_inputs,
                                          self.subtype_inputs, config)
         self.doc_emebedding = self.doc_embedding_layer(self.doc_inputs, self.lstm_dim,
                                          self.length, config)
         lstm_inputs = tf.nn.dropout(self.embedding,1-self.dropout)

         lstm_outputs,lstm_states = self.biLSTM_layer(lstm_inputs,self.lstm_dim,self.length)
         lstm_outputs = tf.nn.dropout(lstm_outputs,1-self.dropout)

         self.sen_att_outputs = self.attention(lstm_outputs)
         self.doc_att_outputs = self.doc_attention(self.doc_emebedding,lstm_states)


     def embedding_layer(self, char_inputs, seg_inputs, subtype_inputs, config, name=None):
         embedding = []
         with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
             embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
             if config["seg_dim"]:
                 with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                     self.seg_lookup = tf.get_variable(
                         name="seg_embedding",
                         shape=[self.num_segs, self.seg_dim],
                         initializer=self.initiaizer)
                     embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
             if config["subtype_dim"]:
                 with tf.variable_scope("subtype_embedding"), tf.device('/cpu:0'):
                     self.subtype_lookup = tf.get_variable(
                         name="subtype_embedding",
                         shape=[self.num_subtypes, self.subtype_dim],
                         initializer=self.initiaizer)
                     embedding.append(tf.nn.embedding_lookup(self.subtype_lookup, subtype_inputs))
             embed = tf.concat(embedding, axis=-1)
         return embed

     def doc_embedding_layer(self,doc_inputs,lstm_dim,lengths,config,name=None):

         def doc_LSTM_layer(inputs,lstm_dim,lengths):
             with tf.variable_scope("doc_BiLSTM",reuse=tf.AUTO_REUSE):
                 lstm_cell = {}
                 for direction in ["doc_forward","doc_backward"]:
                     with tf.variable_scope(direction):
                         lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                             lstm_dim,
                             use_peepholes=True,
                             initializer = self.initiaizer,
                             reuse=tf.AUTO_REUSE,
                             state_is_tuple=True
                         )
                 (outputs,
                  (encoder_fw_final_state,encoder_bw_final_state))=tf.nn.bidirectional_dynamic_rnn(
                     lstm_cell["doc_forward"],
                     lstm_cell["doc_backward"],
                     inputs,
                     dtype=tf.float32,
                     sequence_length = lengths
                  )
                 final_state = tf.concat((encoder_bw_final_state.h,encoder_fw_final_state.h),-1)
                 return final_state

         lstm_states = []
         doc_inputs =  tf.reshape(doc_inputs,[self.batch_size,self.seq_nums,self.num_steps])
         doc_input = tf.unstack(tf.transpose(doc_inputs,[1,0,2]),axis=0)
         for i in range(self.seq_nums):
             with tf.variable_scope("doc_embdding",reuse=tf.AUTO_REUSE),tf.device("/cpu:0"):
                 self.char_doc_lookup = tf.get_variable(
                     name="doc_emebdding",
                     shape=[self.num_chars,self.char_dim],
                     initializer=self.initiaizer
                 )
                 doc_embedding = (tf.nn.embedding_lookup(self.char_doc_lookup,doc_input[i]))
             lstm_state = doc_LSTM_layer(doc_embedding,lstm_dim,lengths)
             lstm_states.append(lstm_state)
         last_states = tf.transpose(lstm_states,[1,0,2])
         last_states = tf.reshape(last_states,[self.batch_size,self.seq_nums,self.lstm_dim*2])
         return last_states

     def biLSTM_layer(self,lstm_inputs,lstm_dim,lengths,name=None):
         with tf.variable_scope('char_BiLSTM',reuse=tf.AUTO_REUSE):
             lstm_cell = {}
             for direction in ["forward","backward"]:
                 with tf.variable_scope(direction):
                     lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                         lstm_dim,
                         use_peepholes=True,
                         initializer= self.initiaizer,
                         state_is_tuple= True
                     )
             (outputs,(encoder_fw_final_state,encoder_bw_final_state))= tf.nn.bidirectional_dynamic_rnn(
                        lstm_cell["forward"],
                        lstm_cell["backward"],
                        lstm_inputs,
                        dtype=tf.float32,
                        sequence_length = lengths
             )
             # 每句话经过当前cell后会得到一个state，
             # 经过多少个cell，就有多少个LSTMStateTuple，即每个cell都会输出一个 tuple(c, h)
             # state中的h跟output 的最后一个时刻的输出是一样的，即：output[:, -1, :] = state[0].h
             final_state = tf.concat((encoder_fw_final_state.h,encoder_bw_final_state.h),-1)
             return tf.concat(outputs,-1),final_state


     def attention(self,lstm_outputs,name=None):

         def bilinear_attention(source,target):
             dim1 = int(source.get_shape()[1])
             seq_size = int(target.get_shape()[1])
             dim2 = int(target.get_shape()[2] )
             with tf.variable_scope("att",reuse=tf.AUTO_REUSE):
                 W = tf.get_variable("att_W",
                    shape=[dim1,dim2],
                    dtype=tf.float32,
                    initializer=self.initiaizer )
                 source = tf.expand_dims(tf.matmul(source,W),1) #[b,1,dim2]
                 prod = tf.matmul(source,target,adjoint_b=True) #[b,1,dim2]*[b,s,dim1]
                 prod = tf.reshape(prod,[-1,seq_size]) #[b,seq_size]
                 prod = tf.tanh(prod)

                 alpha = tf.nn.softmax(prod)
                 probs3dim = tf.reshape(alpha,[-1,1,seq_size])
                 Bout = tf.matmul(probs3dim,target) #[b,1,s]*[b,s,dim2]
                 Bout2dim = tf.reshape(Bout,[-1,dim2])
                 return Bout2dim,alpha

         with tf.variable_scope("attention" if not name else name):
             hidden_dim = self.lstm_dim*2
             sequence_length = self.num_steps
             lstm_outputs = tf.reshape(lstm_outputs,[self.batch_size,self.num_steps,hidden_dim])
             outputs = tf.unstack(tf.transpose(lstm_outputs,[1,0,2]),axis=0)
             fina_outputs = list()
             for i in range(sequence_length):
                 atten_output, P = bilinear_attention(outputs[i],lstm_outputs) #每个词和句子做相似度计算
                 fina_outputs.append(atten_output)

             attention_outputs = tf.transpose(fina_outputs,[1,0,2])
             output = tf.reshape(attention_outputs,[self.batch_size,sequence_length,hidden_dim])
             return output

     def doc_attention(self,doc_embedding,lstm_states,name=None):

         def bilinear_attention(source,target):
             dim1 = int(source.get_shape()[1])
             seq_size = int(target.get_shape()[1])
             dim2 = int(target.get_shape()[2])
             with tf.variable_scope("doc_attention",reuse=tf.AUTO_REUSE):
                 W = tf.Variable(tf.truncated_normal([dim1,dim2],0,1.0),tf.float32,name="W_doc_attnetion")
                 b = tf.Variable(tf.truncated_normal([1],0,1.0),tf.float32,name="b_doc_att")

                 source = tf.expand_dims(tf.matmul(source,W),1) #[b,1,dim2]
                 prod = tf.add(tf.matmul(source,target,adjoint_b=True),b) #[b,1,dim2]*[b,s,dim2]
                 prod = tf.reshape(prod,[-1,seq_size])
                 prod = tf.tanh(prod)
                 alpha = tf.nn.softmax(prod)

                 prob3dim = tf.reshape(alpha,[-1,1,seq_size]) #[b,1,s]
                 Bout = tf.matmul(prob3dim,target) #[b,1,s]*[b,s,dim2]
                 Bout2dim = tf.reshape(Bout,[-1,dim2])
                 return Bout2dim,alpha

         with tf.variable_scope("doc_attention" if not name else name):
             hidden_dim = self.lstm_dim*2
             sequence_length = self.num_steps
             lstm_states = tf.reshape(lstm_states,[self.batch_size,hidden_dim])
             atten_output,p = bilinear_attention(lstm_states,doc_embedding)
             output = tf.reshape(atten_output,[self.batch_size,hidden_dim])
             output = tf.expand_dims(output,1)
             output = tf.tile(output,[1,sequence_length,1])
             return tf.reshape(output,[self.batch_size,self.num_steps,hidden_dim])


if __name__ == '__main__':

     '''
     self.char_dim = config["char_dim"]

     self.lstm_dim = config["lstm_dim"]
     self.seg_dim = config["seg_dim"]
     self.subtype_dim = config["subtype_dim"]
     self.num_tags = config["num_tags"]
     self.num_chars = config["num_char"]
     self.num_steps = config["num_steps"]
     '''
     config = {}
     config["lr"] = 0.001
     config["char_dim"] = 100
     config["lstm_dim"] = 200
     config["seg_dim"] = 20
     config["subtype_dim"] = 20
     config["num_tags"] = 15
     config["num_char"] = 26
     config["num_steps"] = 40 #num_step #seq_size(seq_nums) 8

     # char_inputs = tf.Variable(np.reshape(np.arange(160),[4,40]))#batch_size,seq_length
     # doc_inputs = tf.Variable(np.reshape(np.arange(1280),[4,8,40]))
     # seg_inputs = tf.Variable(np.reshape(np.arange(160),[4,40]))
     # subtype_inputs = tf.Variable(np.reshape(np.arange(160),[4,40]))
     # targets =  tf.Variable(np.reshape(np.arange(160),[4,40]))

     char_inputs = np.random.randint(0,26,(4,40))#batch_size,num_steps
     doc_inputs = np.random.randint(0,26,(4,8,40))#batch_size seq_num,num_steps
     seg_inputs = np.random.randint(0,14,(4,40)) #14
     subtype_inputs = np.random.randint(0,51,(4,40)) #51
     targets =  np.random.randint(0,15,(4,40))

     print(doc_inputs)
     with tf.Session() as sess:
        model = Model(config)
        sess.run(tf.global_variables_initializer())
        feed_dict = {model.char_inputs: char_inputs,
                     model.doc_inputs: doc_inputs,
                     model.seg_inputs: seg_inputs,
                     model.subtype_inputs: subtype_inputs,
                     model.targets: targets,
                     model.dropout:0.5
                     }
        embedding, doc_emebedding,sen_att,doc_att= sess.run([model.embedding,model.doc_emebedding,
                                              model.sen_att_outputs,model.doc_att_outputs], feed_dict=feed_dict)
        print(embedding.shape) # 4*40*400
        print(doc_emebedding.shape) # 4*8*400
        print("sen attention shape ",sen_att.shape)
        print("doc attention sahpe ",doc_att.shape)
        # for v in tf.trainable_variables():
        #     print(v)

