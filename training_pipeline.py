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
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import torch as th

from preprocessing.batching import *
from constants import *

from datetime import datetime

from model import *
from preprocessing.embedding_matrix import get_glove

def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_param_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


class Training:

    def __init__(self):
        self.use_cuda = th.cuda.is_available() and (not DISABLE_CUDA)
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        #TO BE UPDATED
        self.word2id_path = "word_vectors/word2id.csv"
        self.id2word_path = "word_vectors/id2word.csv"
        self.glove_path = "word_vectors/glove.6B.300d.txt"
        self.emb_mat_path = "word_vectors/embedding_matrix.txt"
        self.question_path = "train_dev_files/preprocessed_training_question.txt"
        self.context_path = "train_dev_files/preprocessed_training_context.txt"
        self.ans_path = "train_dev_files/preprocessed_training_ans_span.txt"
        self.model = " "
        self.optimizer = " "
        self.params = " " 
        self.global_step = 0


    def get_mask_from_seq_len(self, seq_mask):
        seq_lens = np.sum(seq_mask, axis=1)
        max_len = np.max(seq_lens)
        indices = np.arange(0, max_len)
        mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
        return mask


    def get_data(self, batch, is_train=True):
        qn_mask = self.get_mask_from_seq_len(batch.qn_mask)
        qn_mask_var = th.from_numpy(qn_mask).long()

        context_mask = self.get_mask_from_seq_len(batch.context_mask)
        context_mask_var = th.from_numpy(context_mask).long()

        qn_seq_var = th.from_numpy(batch.qn_ids).long()
        context_seq_var = th.from_numpy(batch.context_ids).long()

        if is_train:
            span_var = th.from_numpy(batch.ans_span).long()
            span_s, span_e = self.get_spans(span_var)

        if self.use_cuda:
            qn_mask_var = qn_mask_var.cuda()
            context_mask_var = context_mask_var.cuda()
            qn_seq_var = qn_seq_var.cuda()
            context_seq_var = context_seq_var.cuda()
            if is_train:
                span_var = span_var.cuda()
                span_s, span_e = self.get_spans(span_var)

        if is_train:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var, span_s, span_e 
        else:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var

    def get_spans(self, span):
        span_start=th.zeros(span.shape[0])
        span_end=th.zeros(span.shape[0])
        for k in span:
            span_start=k[0]
            span_end=k[1]
        return span_start, span_end

    def seq_to_emb(self, seq):
        seq_list = seq.tolist()
        emb_list = [[self.emb_mat[y] for y in x] for x in seq_list]
        return th.Tensor(emb_list)

    def train_one_batch(self, batch, model, optimizer, params):
        model.train()
        optimizer.zero_grad()
        q_seq, q_lens, d_seq, d_lens, span_s, span_e = self.get_data(batch)

        # convert sequence into embedding
        q_emb = self.seq_to_emb(q_seq).cuda()
        d_emb = self.seq_to_emb(d_seq).cuda()

        print(q_seq.shape, d_seq.shape)
        print(q_emb.shape, d_emb.shape)
        loss, _, _ = model(d_emb, q_emb, span_s, span_e)

        l2_reg = None
        for W in params:
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        loss = loss + REG_LAMBDA * l2_reg

        loss.backward()

        param_norm = get_param_norm(params)
        grad_norm = get_grad_norm(params)

        clip_grad_norm_(params, MAX_GRAD_NORM)
        optimizer.step()
        print(loss.item())
        return loss.item(), param_norm, grad_norm


    def training(self):
        #print("Reading word2id and id2word...", end='')
        #word2id = pd.read_csv(self.word2id_path).to_dict()
        #id2word = pd.read_csv(self.id2word_path).to_dict()
        #print("done.")

        #print("Reading embedding matrix (this takes a while)...", end='')
        #emb_mat = np.loadtxt(self.emb_mat_path)
        #print("done.")

        self.emb_mat, self.word2id, self.id2word = get_glove(self.glove_path, EMBEDDING_DIM) 

        serial_path = "model/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        os.mkdir(serial_path)
        
        for epoch in range(NUM_EPOCHS):
            iter_tic = time.time()
             
            for batch in get_batch_generator(self.word2id, self.context_path, self.question_path, self.ans_path, 64, context_len=MAX_CONTEXT_LEN,
                    question_len=MAX_QUESTION_LEN, discard_long=True):
                # global_step += 1

                # TODO build doc and que matrix
                
                self.model = DCNModel(BATCH_SIZE, self.device) 
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.1, amsgrad=True)
                loss, param_norm, grad_norm = self.train_one_batch(batch, self.model, self.optimizer, self.params)
            iter_toc = time.time()
            iter_time = iter_toc - iter_tic
            print("Epoch %i completed in %i seconds" % (epoch, iter_time))

            # save model

            print("Serialising model parameters ...", end='')
            th.save(self.model.state_dict(), serial_path + "epoch_%i.par" % epoch) 
            print("done.")

Training().training()
