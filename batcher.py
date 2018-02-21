import random

import numpy as np


class Batcher:
    def __init__(self, size, data, shuffle=False, repeat=False, pad=False, seed=1337):
        self.size = size
        self.repeat = repeat
        self.pad = pad
        self.shuffle = shuffle
        self.images = data['image']
        self.questions = data['question']
        self.labels = data['label']
        self.keys = list(self.images.keys())
        random.seed(seed)

        if shuffle:
            random.shuffle(self.keys)

        self.batch_gen = chunks(self.keys, self.size)

    def next_batch(self):
        try:
            keys = next(self.batch_gen)
        except StopIteration:
            if self.shuffle:
                random.shuffle(self.keys)
            if self.repeat:
                self.batch_gen = chunks(self.keys, self.size)
                keys = next(self.batch_gen)
            else:
                return

        images = np.array(collect_keys(self.images, keys))
        max_q_len = max([np.array(x).shape[0] for x in collect_keys(self.questions, keys)])
        questions = np.array([pad_seq(np.array(x).tolist(), max_q_len) for x in collect_keys(self.questions, keys)])
        labels = np.array([np.array(x) for x in collect_keys(self.labels, keys)])

        if len(keys) != self.size and self.pad:
            ref_images = np.zeros(([self.size] + list(images.shape[1:])))
            ref_questions = np.zeros(([self.size] + list(questions.shape[1:])))
            ref_labels = np.zeros(self.size)

            images = pad(images, ref_images, offsets=[0] * len(images.shape))
            questions = pad(questions, ref_questions, offsets=[0] * len(questions.shape))
            labels = pad(labels, ref_labels, offsets=[0] * len(labels.shape))

        return images, questions, labels


def pad(array, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insert_here = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insert_here] = array
    return result


def pad_seq(seq, maxlen, reverse=False):
    """ Pad or shorten a list of items """
    res = seq
    if len(seq) > maxlen:
        if reverse:
            del res[:(len(seq) - maxlen)]
        else:
            del res[maxlen:]
    elif len(seq) < maxlen:
        if reverse:
            res = [0] * (maxlen - len(seq)) + res
        else:
            res.extend([0] * (maxlen - len(seq)))
    return res


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def collect_keys(l, keys):
    return [l[k] for k in keys]
