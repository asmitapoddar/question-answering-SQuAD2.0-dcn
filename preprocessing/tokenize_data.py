from stanfordnlp.server import CoreNLPClient

import numpy as np 
from scipy import spatial 
import matplotlib.pyplot as plt 

from tqdm import tqdm 
import json

# This script produces a tokenised version of the SQuAD dataset


## To run, download Stanford CORENLP and then ... 
## export CORENLP_HOME=/.../.../stanford-corenlp-full-2018-10-05/
## python3 preprocess_data.py

def load_train_set(): 
    with open("data/train-v2.0.json", "r") as f:
        train_set = json.load(f)
    return train_set

def load_dev_set(): 
    with open("data/dev-v2.0.json", "r") as f:
        dev_set = json.load(f)
    return dev_set

def tokenize(client, text):
    # We assume that you've downloaded Stanford CoreNLP and defined an environment
    # variable $CORENLP_HOME that points to the unzipped directory.
    # The code below will launch StanfordCoreNLPServer in the background
    # and communicate with the server to annotate the sentence.
    
    # submit the request to the server
    ann = client.annotate(text)

    tokens = []
    for sentence in ann.sentence: 
        tokens += [[token.word, token.beginChar, token.endChar] for token in sentence.token]
    return tokens

def preprocess(nlp_client, dataset, outFile):
    # Takes the parsed JSON dataset, and replaces the questions and context documents 
    # by a sequence of word embeddings after tokenisation.

    # Array
    data = dataset["data"]
    for item in tqdm(data): 
        for para in item["paragraphs"]:
            # This is the short paragraph of context for the question
            context = para["context"]
            para["context_tokens"] = tokenize(nlp_client, context)

            for qas in para["qas"]:
                # Question text
                question = qas["question"]
                qas["question_tokens"] = tokenize(nlp_client, question)

                # Unique identifier for (question, corresponding answers)
                qas_id = qas["id"]

                is_impossible = qas["is_impossible"]

                # Each question typically has multiple answers
                # Could be empty (if is_impossible = True, new in SQuAD v2)
                for ans in qas["answers"]:
                    answer_text = ans["text"]
                    ans["text_tokens"] = tokenize(nlp_client, answer_text)
    
    # Write to file 
    with open(outFile, "w") as outFile: 
        json.dump(dataset, outFile)

# set up the client
corenlpProps = {}
corenlpProps["tokenize.options"] = "ptb3Escaping=false,invertible=true"
with CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=60000, memory='16G') as client:
    preprocess(client, load_train_set(), "data/train-v2.0-tokenized.json")
    preprocess(client, load_dev_set(), "data/dev-v2.0-tokenized.json")
