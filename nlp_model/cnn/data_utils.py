#! -*- coding: utf-8 -*-
import numpy as np
import json,re,os,pickle
from gensim.models import KeyedVectors

embeddingspath = "../Word2Vec/GoogleNews-vectors-negative300.bin"


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
	# Load data from files
	positive_examples = list(open(positive_data_file, "r", encoding='utf-8',errors='ignore').readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open(negative_data_file, "r", encoding='utf-8',errors='ignore').readlines())
	negative_examples = [s.strip() for s in negative_examples]
	# Split by words
	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]
	#x_text = [s.split(" ") for s in x_text]
    # Generate labels
	positive_labels = [0 for _ in positive_examples]
	negative_labels = [1 for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)
	return [x_text, y]

def create_dico(item_list):
    #对词列表创建字典
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    获取word->id 和id->word字典
    按照词频出现进行排序
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i+2: v[0] for i, v in enumerate(sorted_items)}
    id_to_item[0] = "PAD"
    id_to_item[1] = "UNK"
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


#数据对齐
def pad_sentences(sentences,padding_word="pad"):
	seq_length = max[len(x.split(" ")) for x in sentences]
	print("max seq_length",seq_length)
	padded_sentence = []
	for i,sen in enumerate(sentences):
		x = sen.split(" ")
		num_padding =seq_length-len(sen.split())
		new_sen = x+[padding_word]*num_padding
		padded_sentence.append(new_sen)
	return padded_sentence

def char_mapping(sentences,lower):
    #获取词的列表
	chars = [[x.lower() if lower() else x for x in s.split()] for s in sentence]
    dico = cretae_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" %(len(dico),sum(len(x) for x in chats)     

#获得文本的所有词,建立字典
def build_vocab(sentences):
	#词典
	vocabs = {}
	if not os.path.exists('./all_chars_me.json'):
		for i,sen in enumerate(sentences):
			for word in sen.split(" "):
				vocabs[word] = vocabs.get(word,0)+1
		id2word = {i+2:j for i,j in enumerate(vocabs)} # 0: mask, 1: padding
		id2word[0] = 'pad'
		id2word[1] = 'mask'
		word2id = {j:i for i,j in id2word.items()}
		json.dump([id2word, word2id], open('./all_word_me.json', 'w'))
	
	else:
		id2word,word2id = json.load(open('./all_word_me.json','r'))

	return word2id,id2word

def build_input(sentences,label,vocabs):
	data = [] 
	for sen in sentences:
		x = [vocabs[word if word in vocabs else "pad"]for word in sen]
		data.append(x)
	np.array(data,dtype='int32')
	y = np.array(label,dtype='int32')
	return data,y

def load_word2vec(emb_path, id_to_word, word_dim):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = np.zeors(len(id_to_word),word_dim)
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if(i==0):
            new_wwights[i+1] = np.random.uniform(-0.25,0.25,word_dim)
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:    #replace numbers to zero
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


def load_wv(id2word):
	#加载向量
	c_found = 0
	c_lower = 0
	c_zeros = 0
	n_words = len(id2word)
	pre_trained = KeyedVectors.load_word2vec_format(embeddingspath,binary=True,unicode_errors='ignore')
	wordEmbeddings = np.zeros((len(id2word),300));
	for idx,word in id2word.items():
		if(idx==1):
			wordEmbeddings[idx] = np.random.uniform(-1,1,300)
		if word in pre_trained:
			wordEmbeddings[idx] = pre_trained[word]
			c_found+=1
		elif word.lower() in pre_trained:
			wordEmbeddings[idx] = pre_trained[word.lower()]
			c_lower+=1
		elif re.sub('\d','0',word.lower()) in pre_trained:
			wordEmbeddings[idx] = pre_trained[re.sub('\d','0',word.lower())]
			c_zeros+=1
	print("%i found directly, %i after lowercasing, %i after lowercasing + zero." %(c_found,c_lower,c_zeros))
	return wordEmbeddings
						  
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    生成一个批次的数据.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # 将每一轮的数据打乱
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]						  


if __name__ == '__main__':
	data_folder = ["./data/rt-polarity.pos","./data/rt-polarity.neg"]
	data = load_data_and_labels(data_folder[0],data_folder[1])
	sentences = data[0]
	labels = data[1]
	word2id,id2word = build_vocab(sentences)
	print('word index',len(word2id))
	padd_sentences = pad_sentences(sentences)
	wordEmbeddings = load_wv(id2word)
	x,y = build_input(padd_sentences,labels,word2id)

	data = {"wordEmbeddings":wordEmbeddings,"sen":x,"labels":y}
	pickle.dump(data,open('./data/data.bin','wb'))
