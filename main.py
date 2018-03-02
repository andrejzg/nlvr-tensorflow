import json
import data_ops
import itertools
import batcher
import models
import logging
import experiment_logging
import sys
import shutil
import time

import tensorflow as tf

from collections import defaultdict
from tensorflow.python.training.summary_io import SummaryWriterCache

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

# Read in user-provided config.py and run name.
args = sys.argv[1:]
assert len(args) == 2, 'usage: <path to config> <run_name>'
config_path, run_name = args

# Import the user-provided config.py as a module
user_config = data_ops.import_module(config_path).config

# Logdir naming convention
time_str = time.strftime('%d|%m|%Y@%H:%M:%S')
logdir = f'model_logs/{run_name}_{time_str}'

# Preprocess data
train_raw = [json.loads(x) for x in open('data/train/train.json').readlines()]
test_raw = [json.loads(x) for x in open('data/test/test.json').readlines()]
dev_raw = [json.loads(x) for x in open('data/dev/dev.json').readlines()]

train = data_ops.make_examples(train_raw, split='train')
test = data_ops.make_examples(test_raw, split='test')
dev = data_ops.make_examples(dev_raw, split='dev')

# Build vocab
all_sentences = [x[0] for x in train]
all_words = list(set([x.lower() for x in list(itertools.chain.from_iterable([x.split(' ') for x in all_sentences]))]))
word2id = defaultdict(lambda: 1)  # use 0 for pad, 1 for unk
for i, word in enumerate(all_words, start=2):
    word2id[word] = i
id2word = {i: w for w, i in word2id.items()}

# Prepare config file
config = {
        'model_fn': models.simple_conv_lstm,
        'vocab_size': len(word2id),
        'lr': 5e-3,
        'eval_every_steps': 200,
        'x_resize': 50,
        'y_resize': 200
    }

resize = (config['y_resize'], config['x_resize'])

for k in config:
    if k in user_config:
        config[k] = user_config[k]

# Convert data into HDF5 format
train_numeric = data_ops.load_or_make_numeric_examples(train,
                                                       word2id=word2id,
                                                       split='train',
                                                       save_path='data/train/train.hdf5',
                                                       resize=resize)
test_numeric = data_ops.load_or_make_numeric_examples(test,
                                                      word2id=word2id,
                                                      split='test',
                                                      save_path='data/test/test.hdf5',
                                                      resize=resize)
dev_numeric = data_ops.load_or_make_numeric_examples(dev,
                                                     word2id=word2id,
                                                     split='dev',
                                                     save_path='data/dev/dev.hdf5',
                                                     resize=resize)

# Batcher
batcher_train = batcher.Batcher(data=train_numeric, size=20, shuffle=True, pad=True, repeat=True)
batcher_test = batcher.Batcher(data=test_numeric, size=20)  # size=len(test_numeric['image']))
batcher_dev = batcher.Batcher(data=dev_numeric, size=len(dev_numeric['image']))  # size=len(dev_numeric['image']))

# Graph inputs
images_t = tf.placeholder(tf.float32, [None, *resize, 3], name='images_t')  # The 3 there is the number of RGB channels
questions_t = tf.placeholder(tf.int32, [None, None], name='questions_t')
labels_t = tf.placeholder(tf.int32, [None, ], name='labels_t')

# Model outputs
logits = config['model_fn'](images_t, questions_t, config)
prediction_probs = tf.nn.softmax(logits, axis=-1)
predictions = tf.argmax(logits, axis=-1)

# Loss
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=labels_t)
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=config['lr'])
global_step_t = tf.train.create_global_step()
train_op = optimizer.minimize(loss, global_step=global_step_t)

# Session
sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=60,
        save_summaries_steps=50
        )

# Summaries
summary_writer = SummaryWriterCache.get(logdir)
metrics_logger = experiment_logging.TensorboardLogger(writer=summary_writer)
shutil.copyfile(config_path, logdir + '/config.py')  # save config in logdir

# Fetch entire dev set (no need to do this inside the eval loop repeatedly)
image_dev, question_dev, label_dev = next(batcher_dev)
dev_feed_dict = {
                images_t: image_dev,
                questions_t: question_dev,
                labels_t: label_dev
            }

# Train-Eval loop
while True:
    image, question, label = next(batcher_train)
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
