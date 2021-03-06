import itertools
import string
import h5py
import os
import importlib.util

import numpy as np

from tqdm import tqdm
from PIL import Image

table = str.maketrans({key: None for key in string.punctuation})


def make_examples(raw_examples, split):
    sentence_image_label_triplets = [(x['sentence'], '{}/{}-{}'.format(x['directory'], split, x['identifier']), x['label']) for x in raw_examples]
    combinations = itertools.product(sentence_image_label_triplets, range(0, 5))
    example_pairs = [(clean_text(x[0]), 'data/{}/images/{}-{}.png'.format(split, x[1], y), 1 if x[2] == 'true' else 0) for x, y in combinations]
    return example_pairs


def load_or_make_numeric_examples(examples, word2id, split, save_path, resize=None):
    if os.path.isfile(f'data/{split}/{split}.hdf5'):
        hf = h5py.File(f'data/{split}/{split}.hdf5')
    else:
        make_numeric_examples(examples, split, word2id, save_path, resize)
        hf = h5py.File(f'data/{split}/{split}.hdf5')
    return hf


def make_numeric_examples(examples, split, word2id, save_path, resize=None):
    numeric_examples = []
    skipped = 0
    path = save_path
    hf = h5py.File(path)
    q_group = hf.create_group('question')
    i_group = hf.create_group('image')
    l_group = hf.create_group('label')
    for i, x in enumerate(tqdm(examples, desc=split)):
        sentence, image_path, label = x
        word_ids = [word2id[w] for w in clean_text(sentence).split()]
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            if resize:
                image = image.resize(resize[::-1])
            image = np.array(image)
        except IOError:
            skipped += 1
            pass
        name = image_path.split('/')[-1].split('.')[0]

        q_group.create_dataset(f'{name}', data=word_ids)
        i_group.create_dataset(f'{name}', data=image)
        l_group.create_dataset(f'{name}', data=label)

        numeric_examples.append((word_ids, image, label))
    print(f'{split}: {skipped} record(s) skipped.')
    hf.close()
    print(f'{split}: saved into {path}')
    return hf


def clean_text(text):
    # Remove punctuation and lowercase
    text = text.translate(table)
    text = text.lower()

    # Remove redundant white space
    text = " ".join(text.split())

    return text


def import_module(path):
    spec = importlib.util.spec_from_file_location('', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
