"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

import csv
import numpy as np

from constants import *
#from google.colab import files
from tqdm import tqdm

def get_glove(glove_path, glove_dim):

    print ("Loading GLoVE vectors from file: %s" % glove_path)
    vocab_size = 2196018 # this is the vocab size of the Glove Corpus downloaded (glove.840B.300d.txt)

    emb_matrix = np.zeros((vocab_size + len(START_VOCAB), glove_dim))  #(2196020, 300)
    print(emb_matrix.shape)
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(START_VOCAB), :] = np.zeros(len(START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in START_VOCAB:
        #print('word', idx)
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    """
    # Length of vocab
    with open(glove_path, 'r') as f:
        for i, l in enumerate(f):
            pass
        print('Length',i+1)
    """
    #print('index', idx)

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=20):
            values = line.split()
            word = ''.join(line[:-(len(' '.join(values[-300:]))+2)])
            if word in word2id:
                continue
            #print(word)
            #print(line, word)
            
            vector = list(map(float, values[-300:]))

            if glove_dim != len(vector):
                raise Exception("You set --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size=%i matches!" % (glove_dim, len(vector)))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    assert len(word2id) == len(id2word)

    """
    skipping over bad embeddings
    final_vocab_size = vocab_size + len(START_VOCAB)
    #assert len(word2id) == final_vocab_size
    #assert len(id2word) == final_vocab_size
    #assert idx == final_vocab_size
    """

    return emb_matrix, word2id, id2word
'''
#Write to file

#For dictionary
w = csv.writer(open("word2id.csv", "w"))
for key, val in word2id.items():
    w.writerow([key, val])
#files.download('word2id.csv') #for downloading to Google Drive

w1 = csv.writer(open("id2word.csv", "w"))
for key, val in id2word.items():
    w1.writerow([key, val])
    
#For embedding_matrix
np.savetxt('embedding_matrix.txt', emb_matrix,fmt='%.2f')

#f=open("embedding_matrix.txt", "r")
#embedding_matrix = f.read()
'''
