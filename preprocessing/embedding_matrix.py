
"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1


def get_glove(glove_path, glove_dim):

    print ("Loading GLoVE vectors from file: %s" % glove_path)
    vocab_size = 2196018 # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh):
            values = line.split()
            word = ''.join(line[:-(len(' '.join(values[-300:]))+2)])
            
            vector = list(map(float, values[-300:]))
            if glove_dim != len(vector):
                continue;
                raise Exception("You set --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size=%i matches!" % (glove_dim, len(vector)))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1
            

    final_vocab_size = vocab_size + len(_START_VOCAB)
    #assert len(word2id) == final_vocab_size
    #assert len(id2word) == final_vocab_size
    #assert idx == final_vocab_size

    return emb_matrix, word2id, id2word
