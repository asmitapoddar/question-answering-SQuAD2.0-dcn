"""
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
import sys
sys.path.append('AML_Project/word_vectors/glove.840B.300d.txt')
sys.path.append('/content/gdrive/My Drive/AML_Project/word_vectors/word2id.csv')
sys.path.append('/content/gdrive/My Drive/AML_Project/word_vectors/id2word.csv')
sys.path.append('/content/gdrive/My Drive/AML_Project/Training and development files/preprocessed_training_question.txt')
sys.path.append('/content/gdrive/My Drive/AML_Project/Training and development files/preprocessed_training_context.txt')
sys.path.append('/content/gdrive/My Drive/AML_Project/Training and development files/preprocessed_training_ans_text.txt')

#For me:
word2id_path = sys.path[12]
question_path = sys.path[14]
context_path = sys.path[15]
ans_path = sys.path[16]
"""

import io
import json
import logging
import os
import sys
import time
import numpy as np
import torch
import pandas as pd
import re
from batching import *

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1
reg_lambda = 0.1
max_grad_norm = 0.5

use_cuda = torch.cuda.is_available()

#TO BE UPDATED
word2id_path = " "
question_path = " "
context_path = " "
ans_path = " "
model = " "
optimizer = " "
params = " " 
global_step = 0

# from preprocessing/batching import get_batch_generator

def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids

def get_mask_from_seq_len(self, seq_mask):
    seq_lens = np.sum(seq_mask, 1)
    max_len = np.max(seq_lens)
    indices = np.arange(0, max_len)
    mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
    return mask

def get_data(self, batch, is_train=True):
    qn_mask = self.get_mask_from_seq_len(batch.qn_mask)
    qn_mask_var = torch.from_numpy(qn_mask).long()

    context_mask = self.get_mask_from_seq_len(batch.context_mask)
    context_mask_var = torch.from_numpy(context_mask).long()

    qn_seq_var = torch.from_numpy(batch.qn_ids).long()
    context_seq_var = torch.from_numpy(batch.context_ids).long()

    if is_train:
        span_var = torch.from_numpy(batch.ans_span).long()

    if use_cuda:
        qn_mask_var = qn_mask_var.cuda()
        context_mask_var = context_mask_var.cuda()
        qn_seq_var = qn_seq_var.cuda()
        context_seq_var = context_seq_var.cuda()
        if is_train:
            span_var = span_var.cuda()

    if is_train:
        return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var, span_var
    else:
        return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var

def get_grad_norm(self, parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def get_param_norm(self, parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def train_one_batch(self, batch, model, optimizer, params):
    model.train()
    optimizer.zero_grad()
    q_seq, q_lens, d_seq, d_lens, span = self.get_data(batch)
    loss, _, _ = model(q_seq, q_lens, d_seq, d_lens, span)

    l2_reg = None
    for W in params:
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    loss = loss + reg_lambda * l2_reg

    loss.backward()

    param_norm = self.get_param_norm(params)
    grad_norm = self.get_grad_norm(params)

    clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    print(loss.item())
    return loss.item(), param_norm, grad_norm



def training():
    epoch = 0
    num_epochs = 1000 
    while epoch < num_epochs:
      epoch += 1
      iter_tic = time.time()
      
      word2id = []
      df = pd.read_csv(word2id_path)
      word2id = df.to_dict()
    
      for batch in get_batch_generator(word2id, context_path, question_path, ans_path, 64, context_len=600,
              question_len=30, discard_long=True):
          global_step += 1
          loss, param_norm, grad_norm = train_one_batch(batch, model, optimizer, params)
      iter_toc = time.time()
      iter_time = iter_toc - iter_tic

    loss, param_norm, grad_norm = train_one_batch(batch, model, optimizer, params)
    iter_toc = time.time()
    iter_time = iter_toc - iter_tic
