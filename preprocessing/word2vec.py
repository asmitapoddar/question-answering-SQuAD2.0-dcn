# import word2vec
import numpy as np 
from scipy import spatial 
import matplotlib.pyplot as plt 

from tqdm import tqdm 
import json

# Dimensionality of word vectors in glove.840B.300d (also in glove.6B.300d)
DIMENSIONALITY = 300

def load_embeddings_index(small = False):
    embeddings_index = {}

    if small:
        filePath = "glove.6B.300d.txt"
    else:
        filePath = "glove.840B.300d.txt"

    print("Loading embeddings from " + filePath)

    with open(filePath, 'r', encoding="utf8") as f:
        for line in tqdm(f):
            values = line.split()
            word = ''.join(values[:-DIMENSIONALITY])
            coefs = np.asarray(values[-DIMENSIONALITY:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index
    
if __name__ == '__load_embeddings__':
    load_embeddings_index()
