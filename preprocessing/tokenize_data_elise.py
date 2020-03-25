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
        for token in sentence.token:
            tokens.append(token.word)
    return tokens

def get_token_index(context, context_tokens):
    """
    Given a char index, returns corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        for 0,1,2,3,4 we return 0 and 6,7,8,9,10 we return 1.
    """
    acc = '' # accumulator
    current_token_index = 0 # current token location
    mapping=dict()
    for char_index,char in enumerate(context):
        
      if char!=' ' and char!='\n':
        current_token=context_tokens[current_token_index]
        acc+=char

        if current_token=="``"or current_token=="''":
          current_token="\""

        if acc==current_token:
        
          start_index=char_index-len(acc)+1
          for location in range(start_index, char_index+1):
            mapping[location]=current_token_index
          acc=''
          current_token_index+=1

 
    if current_token_index != len(context_tokens):
        return None
    else:
        return mapping

def preprocess(nlp_client, dataset, type):
    # Takes the parsed JSON dataset, and replaces the questions and context documents 
    # by a sequence of word embeddings after tokenisation.

    # Array
    data = dataset["data"]
    examples=[]
    for item in tqdm(data): 
        for para in item["paragraphs"]:
            # This is the short paragraph of context for the question
            context = para["context"]
            context_tokens = tokenize(nlp_client, context)
            mapping_context=get_token_index(context,context_tokens)

            for qas in para["qas"]:
                # Question text
                question = qas["question"]
                question_tokens = tokenize(nlp_client, question)

                # Unique identifier for (question, corresponding answers)
                qas_id = qas["id"]

                is_impossible = qas["is_impossible"]

                # Each question typically has multiple answers
                # Could be empty (if is_impossible = True, new in SQuAD v2)
                for ans in qas["answers"]:
                    answer_text = ans["text"]
                    answer_text_tokens = tokenize(nlp_client, answer_text)

                    answer_start = ans["answer_start"]
                    answer_end = answer_start + len(answer_text)-1

                    assert answer_start <= answer_end

                    if mapping_context!=None:
                        token_answer_start=mapping_context[answer_start]
                        token_answer_end=mapping_context[answer_end]
                    else:
                        token_answer_start=None
                        token_answer_end=None

                    examples.append((' '.join(context_tokens), \
                     ' '.join(question_tokens), \
                     ' '.join(answer_text_tokens),\
                     ' '.join([str(token_answer_start), str(token_answer_end)])))
               

    with open('data/preprocessed_'+type+'_context.txt', 'wb+') as context_file,  \
         open('data/preprocessed_'+type+'_question.txt', 'wb+') as question_file,\
         open('data/preprocessed_'+type+'_ans_text.txt', 'wb+') as ans_text_file,\
         open('data/preprocessed_'+type+'_ans_span.txt', 'wb+') as ans_span_file :

    
    # Write to files 
        for (context, question, answer, span) in examples:
            context_file.write(context.encode('utf8') + b'\n')
            question_file.write(question.encode('utf8') + b'\n')
            ans_text_file.write(answer.encode('utf8') + b'\n')
            ans_span_file.write(span.encode('utf8') + b'\n')

# set up the client
corenlpProps = {}
corenlpProps["tokenize.options"] = "ptb3Escaping=false,invertible=true"
with CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=60000, memory='16G', properties=corenlpProps) as client:
    preprocess(client, load_train_set(), "train")
    preprocess(client, load_dev_set(), "dev")
