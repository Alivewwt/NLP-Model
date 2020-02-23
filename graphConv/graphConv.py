from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers,regularizers
import tensorflow as tf
import numpy as np

class GraphConvLayer(Layer):
    def __init__(self, num_inputs,
                 num_units,
                 num_labels,
                 in_arcs=True,
                 out_arcs=True,
                 dropout=0.,
                 batch_first=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # initial cell state. We will just provide the layer input as incomings,
        # unless a mask input, initial hidden state or initial cell state was
        # provided.
    
        # Initialize parent layer
        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        self.retain = 1.0 - dropout
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.batch_first = batch_first

        super(GraphConvLayer, self).__init__(**kwargs)

    def build(self, inputs):
        if self.in_arcs:
            self.V_in = self.add_weight(name="V_in",
                                        shape=(self.num_inputs, self.num_units),
                                        initializer='glorot_normal',
                                        trainable=True)

            self.b_in = self.add_weight(name="b_in",
                                        shape=(self.num_labels, self.num_units),
                                        initializer=initializers.Constant(value=0.),
                                        trainable=True)

            self.V_in_gate = self.add_weight(name="V_in_gate",
                                            shape=(self.num_inputs, 1), 
                                            initializer='uniform',
                                            trainable=True
                                            )
            #initializers.Constant(value=1)
            self.b_in_gate = self.add_weight(name="b_in_gate",
                                            shape=(self.num_labels, 1), 
                                            initializer=initializers.Constant(value=1),
                                            trainable=True
                                            )

        if self.out_arcs:
            self.V_out = self.add_weight(name="V_out",
                                        shape=(self.num_inputs, self.num_units), 
                                        initializer='glorot_normal',
                                        trainable=True)

            self.b_out = self.add_weight(name="b_out",
                                        shape=(self.num_labels, self.num_units), 
                                        initializer=initializers.Constant(value=0.),
                                        trainable=True)

            self.V_out_gate = self.add_weight(name="V_out_gate",
                                            shape=(self.num_inputs, 1), 
                                            initializer='uniform',
                                            trainable=True
                                            )

            self.b_out_gate = self.add_weight(name="b_out_gate",
                                            shape=(self.num_labels, 1), 
                                            initializer=initializers.Constant(value=1),
                                            trainable=True 
                                            )

        self.W_self_loop = self.add_weight(name="W_self_l",
                                            shape=(self.num_inputs, self.num_units), 
                                            initializer='glorot_normal',
                                            trainable=True)

        self.W_self_loop_gate = self.add_weight(name="W_self_l_gate",
                                                shape=(self.num_inputs, 1), 
                                                initializer='uniform',
                                                trainable=True, 
                                                )
        super(GraphConvLayer, self).build(inputs)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return input_shape[0], input_shape[1], self.num_units

    def call(self, inputs, mask=None):
       # Retrieve the layer input
        input0,arc_tensor_in,arc_tensor_out,label_tensor_in,label_tensor_out,mask_in,mask_out,mask_loop = inputs
        batch_size = tf.shape(input0)[0]
        seq_len = int(input0.shape[1])
        #print('batch_size',batch_size,'seq_len',seq_len)

        max_degree = 1

        label_tensor_in=tf.reshape(label_tensor_in,[-1,1])
        label_tensor_out=tf.reshape(label_tensor_out,[-1,1])
        mask_in=tf.reshape(mask_in,[-1,1])
        mask_out=tf.reshape(mask_out,[-1,1])
        mask_loop=tf.reshape(mask_loop,[-1,1])
        
        input_ = tf.reshape(input0,[batch_size*seq_len,self.num_inputs])  # [b* t, h]
        
        if self.in_arcs:
            input_in = K.dot(input_, self.V_in)  # [b* t, h] * [h,h] = [b*t, h]
            second_in = K.gather(self.b_in,K.transpose(label_tensor_in)[0])
            in_ = tf.reshape((input_in+second_in),[batch_size, seq_len, 1, self.num_units])
            # compute gate weights
            input_in_gate = K.dot(input_, self.V_in_gate)  # [b* t, h] * [h,h] = [b*t, h]

            second_in_gate = K.gather(self.b_in_gate,K.transpose(label_tensor_in)[0])
            in_gate = tf.reshape((input_in_gate + second_in_gate),[batch_size, seq_len, 1])
            max_degree += 1

        if self.out_arcs:
            input_out = K.dot(input_, self.V_out)  # [b* t, h] * [h,h] = [b* t, h]
            second_out = K.gather(self.b_out,K.transpose(label_tensor_out)[0])
            degr = K.cast(tf.shape(input_out)[0] / batch_size / seq_len, dtype='int32')
            max_degree += degr

            out_ = tf.reshape((input_out + second_out),[batch_size, seq_len, degr, self.num_units])
            # compute gate weights
            input_out_gate = K.dot(input_, self.V_out_gate)  # [b* t, h] * [h,h] = [b* t, h]
            second_out_gate = K.gather(self.b_out_gate,K.transpose(label_tensor_out)[0])
            out_gate =tf.reshape((input_out_gate + second_out_gate),[batch_size, seq_len, degr])

        same_input = K.dot(input0, self.W_self_loop)
        same_input=tf.reshape(same_input,(batch_size,seq_len,-1))
        same_input=tf.reshape(same_input,(batch_size, seq_len,1, tf.shape(self.W_self_loop)[1]))

        same_input_gate = K.dot(input0, self.W_self_loop_gate)
        same_input_gate=tf.reshape(same_input_gate,(batch_size,seq_len))
        same_input_gate=tf.reshape(same_input_gate,(batch_size,seq_len,-1))

        if self.in_arcs and self.out_arcs:
            potentials = K.concatenate([in_, out_, same_input], axis=2)  # [b, t,  mxdeg, h]
            potentials_gate = K.concatenate([in_gate, out_gate, same_input_gate], axis=2)  # [b, t,  mxdeg, h]
            mask_soft = K.concatenate([mask_in, mask_out, mask_loop], axis=1)  # [b* t, mxdeg]

        elif self.out_arcs:
            potentials = K.concatenate([out_, same_input], axis=2)  # [b, t,  2*mxdeg+1, h]
            potentials_gate = K.concatenate([out_gate, same_input_gate], axis=2)  # [b, t,  mxdeg, h]
            mask_soft = K.concatenate([mask_out, mask_loop], axis=1)  # [b* t, mxdeg]

        elif self.in_arcs:
            potentials =K.concatenate([in_, same_input], axis=2)  # [b, t,  2*mxdeg+1, h]
            potentials_gate = K.concatenate([in_gate, same_input_gate], axis=2)  # [b, t,  mxdeg, h]
            mask_soft = K.concatenate([mask_in, mask_loop], axis=1)  # [b* t, mxdeg]

        potentials_ = K.permute_dimensions(potentials,(3, 0, 1, 2))  # [h, b, t, mxdeg]

        potentials_resh = K.reshape(potentials_,(self.num_units,
                                               batch_size * seq_len,
                                               max_degree))  # [h, b * t, mxdeg]


        potentials_r = K.reshape(potentials_gate,(batch_size * seq_len,
                                                  max_degree))  # [h, b * t, mxdeg]
        # calculate the gate
        mask_soft = K.cast(mask_soft,dtype='float32')
        probs_det_ = K.sigmoid(potentials_r) * mask_soft # [b * t, mxdeg]
        potentials_masked = potentials_resh * mask_soft * probs_det_  # [h, b * t, mxdeg]

        potentials_masked_ = K.sum(potentials_masked,axis=2)  # [h, b * t]
      
        #potentials_masked__ = T.switch(potentials_masked_ > 0, potentials_masked_, 0) # ReLU
        potentials_masked_=K.relu(potentials_masked_)
        #potentials_masked__=K.relu(potentials_masked_)
        result_ = K.permute_dimensions(potentials_masked_,(1, 0))   # [b * t, h]
        result_ = K.reshape(result_,(batch_size, seq_len, self.num_units))  # [ b, t, h]
        #result = result_ * mask.dimshuffle(0, 1, 'x')  # [b, t, h]
        return result_

    def compute_mask(self, inputs, masks=None):
        return None