import json
import data_ops
import itertools
import batcher
import models
import logging
import experiment_logging

import tensorflow as tf

from tensorflow.python.training.summary_io import SummaryWriterCache

train_raw = [json.loads(x) for x in open('data/train/train.json').readlines()]
test_raw = [json.loads(x) for x in open('data/test/test.json').readlines()]
dev_raw = [json.loads(x) for x in open('data/dev/dev.json').readlines()]

train = data_ops.make_examples(train_raw, split='train')
test = data_ops.make_examples(test_raw, split='test')
dev = data_ops.make_examples(dev_raw, split='dev')

# Build vocab
all_sentences = [x[0] for x in train + test + dev]
all_words = list(set([x.lower() for x in list(itertools.chain.from_iterable([x.split(' ') for x in all_sentences]))]))
word2id = {w: i for i, w in enumerate(all_words, start=1)}  # leave index 0 for pad
id2word = {i: w for w, i in word2id.items()}

train_numeric = data_ops.load_or_make_numeric_examples(train, word2id, split='train', save_path='data/train/train.hdf5')
test_numeric = data_ops.load_or_make_numeric_examples(test, word2id, split='test', save_path='data/test/test.hdf5')
dev_numeric = data_ops.load_or_make_numeric_examples(dev, word2id, split='dev', save_path='data/dev/dev.hdf5')

# Batcher
batcher_train = batcher.Batcher(data=train_numeric, size=20, shuffle=True, pad=True, repeat=True)
batcher_test = batcher.Batcher(data=test_numeric, size=20)  # size=len(test_numeric['image']))
batcher_dev = batcher.Batcher(data=dev_numeric, size=len(dev_numeric['image']))  # size=len(dev_numeric['image']))

# config
config = {
    'vocab_size': len(word2id),
    'lr': 5e-3,
    'logdir': 'logs',
    'eval_every_steps': 2
}

# Summaries
summary_writer = SummaryWriterCache.get(config['logdir'])
metrics_logger = experiment_logging.TensorboardLogger(writer=summary_writer)

# tf Graph input
images_t = tf.placeholder(tf.float32, [None, 100, 400, 3], name='images_t')
questions_t = tf.placeholder(tf.int32, [None, None], name='questions_t')
labels_t = tf.placeholder(tf.int32, [None, ], name='labels_t')

# initialize model
logits = models.simple_conv_lstm(images_t, questions_t, config)
prediction_probs = tf.nn.softmax(logits, axis=-1)
predictions = tf.argmax(logits, axis=-1)

# loss
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=labels_t)

# optimizer and train op
optimizer = tf.train.AdamOptimizer(learning_rate=config['lr'])
global_step_t = tf.train.create_global_step()
train_op = optimizer.minimize(loss, global_step=global_step_t)

sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=config['logdir'],
        save_checkpoint_secs=200,
        save_summaries_steps=100
        )

train_writer = tf.summary.FileWriter(config['logdir'], graph=sess.graph)

dev_images, dev_questions, dev_labels = batcher_dev.next_batch()

dev_feed_dict = {
    images_t: dev_images,
    questions_t: dev_questions,
    labels_t: dev_labels
}

test_images, test_questions, test_labels = batcher_test.next_batch()

test_feed_dict = {
    images_t: test_images,
    questions_t: test_questions,
    labels_t: test_labels
}


while True:
    image, question, label = batcher_train.next_batch()
    current_step, train_loss, _ = sess.run([global_step_t,
                                            loss,
                                            train_op
                                            ],
                                           feed_dict={images_t: image,
                                                      questions_t: question,
                                                      labels_t: label})
    metrics_logger.log_scalar('train/loss', train_loss.mean(), current_step)

    print('tick')
    if current_step != 0 and current_step % config['eval_every_steps'] == 0:
        logging.info('Evaluating on dev set.')

        test_predictions, test_probs, test_labels, test_loss = sess.run([predictions,
                                                                         prediction_probs,
                                                                         'labels_t:0',
                                                                         loss],
                                                                        feed_dict=dev_feed_dict)

        metrics_logger.log_scalar('accuracy/dev', (test_predictions == test_labels).mean(), current_step)

        print('tock')


# test and finish batcher
# hook-up analytics
