#encoding =utf-8
from bert import tokenization
import numpy as np

tokenizer = tokenization.FullTokenizer(vocab_file='uncased_L-12_H-768_A-12/vocab.txt',
                                       do_lower_case=True)

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
	positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
	negative_examples = [s.strip() for s in negative_examples]
    # Split by words
	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
	positive_labels = [[0, 1] for _ in positive_examples]
	negative_labels = [[1, 0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)
	return [x_text, y]



def prepare_dataset(sentences,max_sen_len,lower=False):
	data = []
	input_ids = []
	mask_ids = []
	segment_ids = []
	
	for s in sentences:
		text = tokenization.convert_to_unicode(s)
		ids,seg_id,mask_id = convert_single_example(char_line = text,max_sen_len = max_sen_len,tokenizer =tokenizer)
		#input_ids.append(ids)
		#mask_ids.append(mask_id)
		#segment_ids.append(seg_id)
		data.append([ids,mask_id,seg_id])
		
	#data.append([input_ids,mask_ids,segment_ids])

	return data

def convert_single_example(char_line,max_sen_len,tokenizer):
	'''
	对一个样本进行分析，转化成id
	'''
	tokens = []
	text_list = char_line.split(' ')
	for i,tok in enumerate(text_list):
		token = tokenizer.tokenize(tok)
		tokens.extend(token)

	if len(tokens)>=max_sen_len-1:
		tokens =tokens[:max_sen_len-2]
	ntokens = []
	segment_ids =[] 
	ntokens.append("[CLS]")
	segment_ids.append(0)
	for tok in tokens:
		ntokens.append(tok)
		segment_ids.append(0)

	ntokens.append("[SEP]")
	segment_ids.append(0)

	input_ids = tokenizer.convert_tokens_to_ids(ntokens)
	input_mask = [1]*len(ntokens)

	while len(input_ids)<max_sen_len:
		input_ids.append(0)
		segment_ids.append(0)
		input_mask.append(0)

	return input_ids,segment_ids,input_mask

def batch_iter(x_data,y,bacth_size):
	#inputs,masks,segments = x_data[0][0],x_data[0][1],x_data[0][2]
	#x_data = np.array(inputs)
	#data_len = x_data.shape[0]
	x_data =x_data
	y = np.array(y)
	data_len = len(x_data)
	print(data_len)

	num_batchs = int((data_len-1)/bacth_size)+1

	for idx in range(num_batchs):
		start_idx = bacth_size*idx
		end_idx = min(bacth_size*(idx+1),data_len)
		#yield np.array(inputs[start_idx:end_idx]),np.array(masks[start_idx:end_idx]),np.array(segments[start_idx:end_idx]),np.array(y[start_idx:end_idx])
		yield arrange_batch(x_data[start_idx:end_idx]),y[start_idx:end_idx]

def arrange_batch(batch_x):
	input_ids = []
	mask_ids = []
	segment_ids = []
	for ids,mask_id, segment_id in batch_x:
		input_ids.append(ids)
		mask_ids.append(mask_id)
		segment_ids.append(segment_id)
	return input_ids,mask_ids,segment_ids


if __name__ == '__main__':
	x,y = load_data_and_labels("./data/rt-polaritydata/rt-polarity.pos","./data/rt-polaritydata/rt-polarity.neg")
	x = x[:64]
	y = y[:64]
	x_data = prepare_dataset(x,max_sen_len=56,lower=False)
	for batch_x,batch_y in batch_iter(x_data,y,16):
		input_ids,mask_ids,segment_ids = batch_x
		print(len(input_ids))
		print(input_ids)

		
		

