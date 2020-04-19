## tensorflow_model

nlp model implemented by tensorflow

### Text Classification(文本分类)

#### CNN

该模块使用CNN网络对文本进行分类，卷积核大小为[2,3,4]，然后进行最大池化，最后送入到全连接层中进行分类；

#### Bi-LSTM

该模块使用bi-lstm（双向长短时记忆神经网络）对文本进行分类，获得文本时间序列，取最后位置的时间步作为分类向量；

#### HAN

该模块使用GRU对句子和篇章进行编码，然后使用Attention(注意力)机制融合两者的向量，对文本进行分类。

#### BERT

该模块使用目前最火的预训练模型BERT，在下游的分类任务进行微调。

### Doc Embedding(篇章向量)

该模块使用bi-lstm对句子和文档进行建模，获得句子和文档向量，然后进行各自任务的训练。

### GraphConv(图卷积网络)

该模块是图卷积网络的一种实现。

### Sep2Sep(序列到序列)

该模块使用seq2seq对文本标题生成任务进行建模。

### textSimilarity(文本相似度)

该模块针对文本相似度任务进行建模。