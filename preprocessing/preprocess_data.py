from stanfordnlp.server import CoreNLPClient

import numpy as np 
from scipy import spatial 
import matplotlib.pyplot as plt 

from tqdm import tqdm 
import json

## To run, download Stanford CORENLP and then ... 
## export CORENLP_HOME=/.../.../stanford-corenlp-full-2018-10-05/
## python3 preprocess_data.py

# Dimensionality of word vectors in glove.840B.300d (also in glove.6B.300d)
DIMENSIONALITY = 300
print('hola')

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
            word = line[:-(len(' '.join(values[-DIMENSIONALITY:]))+2)]
            coefs = np.asarray(values[-DIMENSIONALITY:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def load_train_set(): 
    with open("train-v2.0.json", "r") as f:
        train_set = json.load(f)
    return train_set

def load_dev_set(): 
    with open("dev-v2.0.json", "r") as f:
        dev_set = json.load(f)
    return dev_set

# dev_set = load_question_answering_dataset()
# print(json.dumps(dev_set, indent=4, sort_keys=True))

def encode_word(word, embeddings):
    # "Set embeddings for out of vocabulary words to zero"
    if word in embeddings: 
        return embeddings[word]
    else:
        return np.zeros(DIMENSIONALITY)

def encode_token_list(embeddings, token_list):
    # Output the sequence of word vectors corresponding to the words in the question/document.
    # i.e. the sequences (x_1^{Q/D}, ..., x_n^{Q/D}) in Section 2.1

    # Each word vector is a column of this matrix
    # word_vectors = np.zeroes(DIMENSIONALITY, len(token_list))
    word_vectors = np.array([encode_word(token, embeddings) for token in token_list])
    return word_vectors.transpose()


def tokenize(client, text):
    # We assume that you've downloaded Stanford CoreNLP and defined an environment
    # variable $CORENLP_HOME that points to the unzipped directory.
    # The code below will launch StanfordCoreNLPServer in the background
    # and communicate with the server to annotate the sentence.
    
    # submit the request to the server
    ann = client.annotate(text)

    tokens = []
    for sentence in ann.sentence: 
        tokens += [token.word for token in sentence.token]
    return tokens

def embed_string(client, embeddings, text):
    tokens = tokenize(client, text)
    # Turn np.ndarrays into nested lists so that it can be JSONified
    return encode_token_list(embeddings, tokens).tolist()

def preprocess(nlp_client, embeddings, dataset):
    # Takes the parsed JSON dataset, and replaces the questions and context documents 
    # by a sequence of word embeddings after tokenisation.

    # Array
    data = dataset["data"]
    print("Computing embeddings for the text in SQuAD")
    for item in tqdm(data): 
        for para in item["paragraphs"]:
            
            # This is the short paragraph of context for the question
            context = para["context"]
            context_embeddings = embed_string(nlp_client, embeddings, context)
            para["context_embedding"] = context_embeddings

            for qas in para["qas"]:
                # Question text
                
                question = qas["question"]
                question_embeddings = embed_string(nlp_client, embeddings, question)
                qas["question_embedding"] = question_embeddings

                # Unique identifier for (question, corresponding answers)
                qas_id = qas["id"]

                is_impossible = qas["is_impossible"]

                # Each question typically has multiple answers
                # Could be empty (if is_impossible = True, new in SQuAD v2)
                for ans in qas["answers"]:
                    
                    answer_text = ans["text"]
                    answer_embeddings = embed_string(nlp_client, embeddings, answer_text)
                    ans["answer_text_embedding"] = answer_embeddings
                    answer_start = ans["answer_start"]
                    #print(answer_start)

    # Write to file 
    with open("preprocessed_training_set.txt", "w") as outFile: 
        json.dump(dataset, outFile)

# set up the client
with CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=60000, memory='16G') as client:
    embeddings = load_embeddings_index(small=False)
    preprocess(client, embeddings, load_train_set())

