import tensorflow as tf
import numpy as np
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("lr", 1e-5, "learning rate for model")
tf.flags.DEFINE_string("save_dir","model","model ckpt save dir")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
	# Load data
	print("Loading data...")
	x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

	# Build vocabulary
	max_document_length = max([len(x.split(" ")) for x in x_text])
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = np.array(list(vocab_processor.fit_transform(x_text)))

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(y)))
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# Split train/test set
	# TODO: This is very crude, should use cross-validation
	dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

	del x, y, x_shuffled, y_shuffled

	print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
	print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
	return x_train, y_train, vocab_processor, x_dev, y_dev


class ModelTrain(object):
	def __init__(self,x_train, y_train, vocab_processor, x_dev, y_dev,is_restore=False):
		self.x_train = x_train
		self.y_train = y_train
		self.x_dev = x_dev
		self.y_dev = y_dev
		self.vocab_processor = vocab_processor
		self.is_restore = is_restore

	def build_graph(self):
		with tf.device("/cpu:0"):
			with tf.Graph().as_default():
				# allow_soft_placement = True: 如果你指定的设备不存在，允许TF自动分配设备
				config = tf.ConfigProto(allow_soft_placement = True)
				#控制GPU资源

				# config.gpu_options_allow_growth = True
				with tf.Session(config=config).as_default() as self.sess:
					self.model = TextCNN(sequence_length=self.x_train.shape[1],
						                num_classes=self.y_train.shape[1],
						                vocab_size=len(self.vocab_processor.vocabulary_),
						                embedding_size=FLAGS.embedding_dim,
						                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
						                num_filters=FLAGS.num_filters,
						                l2_reg_lambda=FLAGS.l2_reg_lambda)

					self.global_step = tf.Variable(0,name="global_step",trainable=False)
					tars = tf.trainable_variables()
					grads,_ = tf.clip_by_global_norm(tf.gradients(self.model.loss,tars),10)
					opt = tf.train.AdamOptimizer(FLAGS.lr)
					self.train_opt = opt.apply_gradients(zip(grads,tars),global_step=self.global_step)
					self.saver = tf.train.Saver(max_to_keep=15)
					self.sess.run(tf.global_variables_initializer())

	def train_step(self,x_batch,y_batch):
		feed_dict = {
			self.model.input_x: x_batch,
			self.model.input_y: y_batch,
			self.model.dropout_rate: FLAGS.dropout_keep_prob
		}
		_, step,  loss, accuracy = self.sess.run(
			[self.train_opt, self.global_step, self.model.loss, self.model.accuracy],
			feed_dict)
		return loss,accuracy

	def dev_step(self,x_batch,y_batch):
		feed_dict = {
			self.model.input_x: x_batch,
			self.model.input_y: y_batch,
			self.model.dropout_keep_prob: 1.0
		}
		loss, accuracy = self.sess.run(
			[self.model.loss, self.model.accuracy],feed_dict)
		return loss,accuracy

	def train(self):
		batches = data_helpers.batch_iter(list(zip(self.x_train, self.y_train)), FLAGS.batch_size, FLAGS.num_epochs)
		losses,accuracy = [],[]
		k = 0
		best_loss = 10000
		stop_tag = False
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			loss,acc = self.train_step(x_batch, y_batch)
			losses.append(loss)
			accuracy.append(acc)
			current_step = tf.train.global_step(self.sess, self.global_step)
			if current_step % FLAGS.evaluate_every == 0:
				print("Evaluation:")
				print("loss:%s,accuracy:%s" %(np.mean(losses),np.mean(accuracy)))
				losses ,accuracy=[],[]
				path = self.saver.save(self.sess, FLAGS.save_dir + "/step_mode.ckpt", current_step)
				print("Saved model checkpoint to {}\n".format(path))
			if current_step % FLAGS.checkpoint_every == 0:
				dev_loss,dev_accuracy = self.dev_step(x_dev,y_dev)
				if dev_loss < best_loss:
					best_loss = dev_loss
					ckpt = self.saver.save(self.sess, FLAGS.save_dir + "/best/step_mode.ckpt",current_step)
					print("Saved best model checkpoint to {}\n".format(ckpt))
					k = 0
				else:
					k+=1
					if k>20:
						stop_tag = True
			if stop_tag:
				break


if __name__ == '__main__':
	x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
	mt = ModelTrain(x_train, y_train, vocab_processor, x_dev, y_dev)
	mt.build_graph()
	mt.train()
