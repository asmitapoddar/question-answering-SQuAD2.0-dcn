from stanfordnlp.server import CoreNLPClient

import numpy as np 
from scipy import spatial 
import matplotlib.pyplot as plt 

from tqdm import tqdm 
import json

## To run, download Stanford CORENLP and then ... 
## export CORENLP_HOME=/.../.../stanford-corenlp-full-2018-10-05/
## python3 preprocess_data.py



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

def preprocess(client, dataset, type):
    # Takes the parsed JSON dataset, and replaces the questions and context documents 
    # by a sequence of word embeddings after tokenisation.

    # Array
    data = dataset["data"]
    print("Computing embeddings for the text in SQuAD")
    examples=[]
    for item in tqdm(data): 
        for para in item["paragraphs"]:
            
            # This is the short paragraph of context for the question
            context = para["context"]
            context_tokens= tokenize(client, context)
            para["context_tokens"]=context_tokens

            for qas in para["qas"]:

                # Question text
                question = qas["question"]
                question_tokens = tokenize(client, question)
                qas["question_tokens"] = question_tokens

                # Unique identifier for (question, corresponding answers)
                qas_id = qas["id"]

                is_impossible = qas["is_impossible"]

                # Each question typically has multiple answers
                # Could be empty (if is_impossible = True, new in SQuAD v2)
                for ans in qas["answers"]:
                    
                    answer_text = ans["text"]
                    answer_tokens = tokenize(client, answer_text)
                    ans["answer_text_tokens"] = answer_tokens
                    answer_start = ans["answer_start"]
                    example.append((context_tokens,question_tokens,answer_tokens))

                    #print(answer_start)
    with open('preprocessed_'+type+'_context.txt', 'w') as context_file,  \
         open('preprocessed_'+type+'_question.txt', 'w') as question_file,\
         open('preprocessed_'+type+'_ans_text.txt', 'w') as ans_text_file)
    
    # Write to files 
    for (context, question, answer) in examples:
          context_file.write(context.encode('utf8') + '\n')
          question_file.write(question.encode('utf8') + '\n')
          ans_text_file.write(answer_text.encode('utf8') + '\n')


def main(client):
    
    # print "Will download SQuAD datasets to {}".format(outFile)
     #print "Will put preprocessed SQuAD datasets in {}".format(outFile)

    # download train set
     train_data = load_train_set()
     print "Train data has %i examples total" % total_exs(train_data)

     # preprocess train set and write to file
     preprocess(client,train_data, 'training')

     # download dev set
     dev_data= load_dev_set()
     print "Dev data has %i examples total" % total_exs(dev_data)

     # preprocess dev set and write to file
     preprocess(client ,dev_data, 'developing')


if __name__ == '__main__':
    
    # set up the client
    with CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=60000, memory='16G') as client:
      main(client)