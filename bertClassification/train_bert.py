#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from bertText import bertText
from process import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    logger.info("max sen length:{}".format(max_document_length))
    x_text = np.array(x_text)
    y = np.array(y)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x_text[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x_text, y, x_shuffled, y_shuffled
    x_train = prepare_dataset(x_train,max_sen_len=max_document_length,lower=False)
    x_dev = prepare_dataset(x_dev,max_sen_len=max_document_length,lower=False)

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev

def train(x_train, y_train, x_dev, y_dev):
    # Training
    # ==================================================
    logger.info("start training")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            #num_classes,max_sen_len,hidden_size,drop_out_keep,
            model = bertText(
                num_classes=2,
                max_sen_len=56,
                hidden_size = 256,
                )
            # Define Training procedure
            '''
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(5e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            '''
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(batch_inputs,batch_masks,batch_segments, batch_y):
                """
                A single training step
                """
                feed_dict = {
                  model.input_ids: np.array(batch_inputs),
                  model.mask_ids: np.array(batch_masks),
                  model.segment_ids:np.array(batch_segments),
                  model.labels: np.array(batch_y),
                  model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [model.opt, model.global_step, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                return loss,accuracy
                
            def dev_step(batch_inputs,batch_masks,batch_segments,batch_y):
                """
                A single training step
                """
                feed_dict = {
                  model.input_ids: np.array(batch_inputs),
                  model.mask_ids: np.array(batch_masks),
                  model.segment_ids:np.array(batch_segments),
                  model.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # Generate batches
            #batches = data_helpers.batch_iter(
                #list(zip(x_train, y_train)), FLAGS.batch_size)
            # Training loop. For each batch...
            losses = []
            accuracy = []
            for epoch in range(FLAGS.num_epochs):
                for batch_x,batch_y in batch_iter(x_train,y_train,FLAGS.batch_size):
                    inputs,masks,segments = batch_x
                    loss,acc = train_step(inputs,masks,segments,batch_y)
                    losses.append(loss)
                    accuracy.append(acc)
                    current_step = tf.train.global_step(sess,model.global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        logger.info("\nEvaluation:")
                        logger.info("loss:{},accuracy:{}".format(np.mean(losses),np.mean(accuracy)))
                        losses = []
                        accuracy = []
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logger.info("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, x_dev, y_dev = preprocess()
    train(x_train, y_train, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()