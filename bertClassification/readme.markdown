### BERT for Text classification

该示例简单的展示使用bert进行文本(句子)分类。主要将bert替换以前的embedding 层，然后得到句子的向量表示，体现bert强大的特征抽取能力。示例中使用的bert预训练模型是[bert_uncase(base)](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)，即不区分大小写，统一成小写。

#### 模型结构

1. bertText.py 是模型的结构文件;
2. process.py 是数据预处理文件，主要将数据格式转化成bert的输入格式：input_ids，segment_ids，mask_ids;
3. train_bert.py 是训练bert的文本分类模型，并将模型保存。



