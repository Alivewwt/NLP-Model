# encoding=utf-8
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from bert import tokenization
import pickle as pkl


tokenizer = tokenization.FullTokenizer(vocab_file="chinese_L-12_H-768_A-12/vocab.txt",
									   do_lower_case=True)


class TripletTextDataset(object):

	def __init__(self, text_a_list, text_b_list, text_c_list, label_list=None):
		if label_list is None or len(label_list) == 0:
			label_list = [None] * len(text_a_list)
		assert all(len(label_list) == len(text_list) for text_list in [text_a_list, text_b_list, text_c_list])
		self.text_a_list = text_a_list
		self.text_b_list = text_b_list
		self.text_c_list = text_c_list
		self.label_list = [0 if label == 'B' else 1 for label in label_list]

	def __len__(self):
		return len(self.label_list)

	def __getitem__(self, index):
		text_a, text_b, text_c, label = self.text_a_list[index], self.text_b_list[index], self.text_c_list[index], \
										self.label_list[index]
		return text_a, text_b, text_c, label

	@classmethod
	def from_dataframe(cls, df):
		text_a_list = df['A'].tolist()
		text_b_list = df['B'].tolist()
		text_c_list = df['C'].tolist()
		if 'label' not in df:
			df['label'] = 'B'
		label_list = df['label'].tolist()
		return cls(text_a_list, text_b_list, text_c_list, label_list)

	@classmethod
	def from_dict_list(cls, data, use_augment=False):
		df = pd.DataFrame(data)
		if 'label' not in df:
			df['label'] = 'B'
		if use_augment:
			df = TripletTextDataset.augment(df)
		return cls.from_dataframe(df)

	@classmethod
	def from_jsons(cls, json_lines_file, use_augment=False):
		with open(json_lines_file, encoding='utf-8') as f:
			data = list(map(lambda line: json.loads(line), f))
		return cls.from_dict_list(data, use_augment)

	@staticmethod
	def augment(df):
		df_cp1 = df.copy()
		df_cp1['B'] = df['C']
		df_cp1['C'] = df['B']
		df_cp1['label'] = 'C'

		df_cp2 = df.copy()
		df_cp2['A'] = df['B']
		df_cp2['B'] = df['A']
		df_cp2['label'] = 'B'

		df_cp3 = df.copy()
		df_cp3['A'] = df['B']
		df_cp3['B'] = df['C']
		df_cp3['C'] = df['A']
		df_cp3['label'] = 'C'

		df_cp4 = df.copy()
		df_cp4['A'] = df['C']
		df_cp4['B'] = df['A']
		df_cp4['C'] = df['C']
		df_cp4['label'] = 'C'

		df_cp5 = df.copy()
		df_cp5['A'] = df['C']
		df_cp5['B'] = df['C']
		df_cp5['C'] = df['A']
		df_cp5['label'] = 'B'

		df = pd.concat([df, df_cp1, df_cp2, df_cp3, df_cp4, df_cp5])
		df = df.drop_duplicates()
		df = df.sample(frac=1)

		return df


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, text_a, text_b=None, text_c=None, label=None):
		"""Constructs a InputExample.
        Args:
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
		self.text_a = text_a
		self.text_b = text_b
		self.text_c = text_c
		self.label = label

	@staticmethod
	def _text_pair_to_feature(text_a, text_b, tokenizer, max_seq_length):
		tokens_a = tokenizer.tokenize(text_a) #分词
		tokens_b = None

		if text_b:
			tokens_b = tokenizer.tokenize(text_b)
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		return input_ids, input_mask, segment_ids

	def to_two_pair_feature(self, tokenizer, max_seq_length):
		ab = self._text_pair_to_feature(self.text_a, self.text_b, tokenizer, max_seq_length)
		ac = self._text_pair_to_feature(self.text_a, self.text_c, tokenizer, max_seq_length)
		ab, ac = InputFeatures(*ab), InputFeatures(*ac)
		return ab, ac


def _truncate_seq_pair(tokens_a: list, tokens_b: list, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop(0)
		else:
			tokens_b.pop(0)

def prepare_dataset(data,max_seq_length):

	text_abs = []
	text_acs = []
	labels = []
	ab_input_ids = []
	ab_input_masks = []
	ab_input_segments = []
	ac_input_ids = []
	ac_input_masks = []
	ac_input_segments = []

	text_a,text_b,text_c,text_label  = data.text_a_list,data.text_b_list,data.text_c_list,data.label_list
	i = 0
	for a,b,c,label in zip(text_a,text_b,text_c,text_label):
		inputex = InputExample(a,b,c,label)
		ab,ac = inputex.to_two_pair_feature(tokenizer,max_seq_length)

		ab_input_ids.append(ab.input_ids)
		ab_input_masks.append(ab.input_mask)
		ab_input_segments.append(ab.segment_ids)

		ac_input_ids.append(ac.input_ids)
		ac_input_masks.append(ac.input_mask)
		ac_input_segments.append(ac.segment_ids)
		i+=1
		if i%100==0:
			logger.info("has been finished %d examples"%(i))

	text_abs+=[ab_input_ids,ab_input_masks,ab_input_segments]
	text_acs+=[ac_input_ids,ac_input_masks,ac_input_segments]

	return text_abs,text_acs,np.eye(2)[labels]

def batch_iter(text_ab, text_ac,labels,batch_size=16):
	assert len(text_ab) == len(text_ac)
	text_ab = np.array(text_ab)
	text_ac = np.array(text_ac)
	print(text_ab.shape)
	data_size = text_ab.shape[1]
	num_batchs = int((data_size-1)//batch_size)+1

	for i in range(num_batchs):
		start_idx = batch_size*i
		end_idx = min(data_size,(i+1)*batch_size)
		yield text_ab[0][start_idx:end_idx],text_ab[1][start_idx:end_idx],text_ab[2][start_idx:end_idx],\
			  text_ac[0][start_idx:end_idx],text_ac[1][start_idx:end_idx],text_ac[2][start_idx:end_idx],\
			  labels[start_idx :end_idx]


def load_datatset(n_splits,dataset_path,test_input_path,test_ground_truth_path):
	data = []

	if n_splits == 1:
		train_data = TripletTextDataset.from_jsons(dataset_path, use_augment=True)
		test_data = TripletTextDataset.from_jsons(test_input_path)
		with open(test_ground_truth_path) as f:
			test_label_list = [line.strip() for line in f.readlines()]
		data.append((train_data,test_data,test_label_list))
		return data

	return data



if __name__ == '__main__':
	datas = load_datatset(1, "./data/train/input.txt", "./data/test/input.txt", './data/test/ground_truth.txt')
	train_data,test_data ,labels = datas[0]
	print(len(train_data),len(test_data),len(labels))

	# with open("./data/train/train.json",'w') as f:
	# 	json.dump([train_data.text_a_list,train_data.text_b_list,
	# 			   train_data.text_c_list,train_data.label_list],f,indent=4,ensure_ascii=False)


	text_abs,text_acs,train_labels = prepare_dataset(train_data,256)
	with open("./data/train/train.pkl","wb") as f:
		pkl.dump([text_abs,text_acs,train_labels],f)

	for batch in  batch_iter(text_abs,text_acs,train_labels,16):
		ab_input_id, ab_input_mask, ab_seg_ids, ac_input_id, ac_input_mask, ac_seg_ids,label = batch
		print(ab_input_id.shape)
		print(ab_input_id)
		print(label)