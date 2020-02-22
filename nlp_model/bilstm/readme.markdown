### tensorflow中RNN

#### tf.contrib.rnn.bidirectional_dynamic_rnn()

在tensorflow中已经提供了双向RNNs的借口，使用tf.contrib.rnn.bidirectional_dynamic_rnn()这个函数，就可以很方便的构建双向RNN网络

首先来熟悉一下接口的参数

```python
bidirectional_dynamic_rnn(
cell_bw,#前向 rnn cell
cell_fw, #后向 rnn cell
inputs, #输入序列
sequence_length = None, #输入长度
initial_state_fw = None, #前向rnn_cell的初始状态 （可选）
initial_state_bw = None, #后向rnn_cell的初始状态 （可选）
dtype = None, #初始化和输出的数据类型
parallel_iterations = None,  
swap_memory = False,
time_major = False# false ,输入格式为[batch_size,seq_len,dim],#true，输入格式为[seq_len,batch_size,dim]
)
```

函数的返回值：

一个(outputs, outputs_state)的一个元组。

若time_major=false，则两个tensor的shape为[batch_size, max_time, depth]，应用在文本中时，max_time可以为句子的长度（一般以最长的句子为准，短句需要做padding），depth为输入句子词向量的维度。

最终的outputs需要使用tf.concat(outputs, 2)将两者合并起来。

outputs_state = (outputs_state_fw， output_state_bw),包含了前向和后向最后的隐藏状态的组成的元组。outputs_state_fw和output_state_bw的类型都是LSTMStateTuple。LSTMStateTuple由(c, h)组成，分别代表memory cell和hidden state

如果还需要用到最后的输出状态，则需要对（outputs_state_fw， output_state_bw）处理:

```python
final_state_c = tf.concat((outputs_state_fw.c, outputs_state_bw.c), 1)
final_state_h = tf.concat((outputs_state_fw.h, outputs_state_bw.h), 1)
outputs_final_state = tf.contrib.rnn.LSTMStateTuple(c=final_state_c,
                                                    h=final_state_h)
```