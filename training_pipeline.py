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
import pathlib

from constants import *
from datetime import datetime
from model import *
from preprocessing.batching import *
from preprocessing.embedding_matrix import get_glove

from torch.nn.utils import clip_grad_norm_

SERIALISATION_KEY_EPOCH = 'epoch'
SERIALISATION_KEY_MODEL = 'model'
SERIALISATION_KEY_OPTIM = 'optim'


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_mask_from_seq_len(seq_mask):
    seq_lens = np.sum(seq_mask, axis=1)
    max_len = np.max(seq_lens)
    indices = np.arange(0, max_len)
    mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
    return mask


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
        self.device = th.device("cuda:0" if th.cuda.is_available() and (not DISABLE_CUDA) else "cpu")
        
        self.model = None
        self.optimizer = None

        #TO BE UPDATED
        self.word2id_path = "word_vectors/word2id.csv"
        self.id2word_path = "word_vectors/id2word.csv"
        self.glove_path = "word_vectors/glove.6B.300d.txt"
        self.emb_mat_path = "word_vectors/embedding_matrix.txt"
        self.question_path = "preprocessing/data/preprocessed_train_question.txt"
        self.context_path = "preprocessing/data/preprocessed_train_context.txt"
        self.ans_path = "preprocessing/data/preprocessed_train_ans_span.txt"
                
        self.params = " " # What's this?
        self.global_step = 0


    def get_data(self, batch, is_train=True):
        qn_mask = get_mask_from_seq_len(batch.qn_mask)
        qn_mask_var = th.from_numpy(qn_mask).long().to(self.device)

        context_mask = get_mask_from_seq_len(batch.context_mask)
        context_mask_var = th.from_numpy(context_mask).long().to(self.device)

        qn_seq_var = th.from_numpy(batch.qn_ids).long().to(self.device)
        context_seq_var = th.from_numpy(batch.context_ids).long().to(self.device)

        if is_train:
            #span_var = th.from_numpy(batch.ans_span).long().to(self.device)
            span_s, span_e = self.get_spans(batch.ans_span)
        
        if is_train:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var, span_s, span_e 
        else:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var

    def get_spans(self, span):
        def to_th(x):
            return th.tensor(list(x)).long().to(self.device)
        
        span_s = to_th(map(lambda p: p[0], span))
        span_e = to_th(map(lambda p: p[1], span))
        return span_s, span_e

    def seq_to_emb(self, seq):
        seq_list = seq.tolist()
        emb_list = [[self.emb_mat[y] for y in x] for x in seq_list]
        return th.tensor(emb_list, dtype=th.float32, device=self.device)

    def train_one_batch(self, batch, model, optimizer, params):
        model.train()
        optimizer.zero_grad()
        q_seq, q_lens, d_seq, d_lens, span_s, span_e = self.get_data(batch)
        
        # convert sequence into embedding
        q_emb = self.seq_to_emb(q_seq)
        d_emb = self.seq_to_emb(d_seq)

        #print("a", q_seq.shape, d_seq.shape)
        #print("b", q_emb.shape, d_emb.shape)
        loss, _, _ = model(d_emb, q_emb, span_s, span_e)

        """
        print("params they not missing!", len(params))
        l2_reg = 0.0 
        for W in params:
            l2_reg = l2_reg + W.norm(2)
        loss = loss + REG_LAMBDA * l2_reg
        """

        #TODO fix L2 Regularisation
        param_norm = get_param_norm(params)
        grad_norm = get_grad_norm(params)

        clip_grad_norm_(params, MAX_GRAD_NORM)
       
        print("loss (incl. reg)", loss)

        loss.backward()
        optimizer.step()

        model.decoder.hmn_alpha.detach_params()
        model.decoder.hmn_beta.detach_params()

        print(loss.item())
        return loss.item(), param_norm, grad_norm


    # Pass state_file_path to resume training from an existing checkpoint.
    def training(self, state_file_path=None):
        """
        print("Reading word2id and id2word...", end='')
        word2id = pd.read_csv(self.word2id_path).to_dict()
        id2word = pd.read_csv(self.id2word_path).to_dict()
        print("done.")

        print("Reading embedding matrix (this takes a while)...", end='')
        emb_mat = np.loadtxt(self.emb_mat_path)
        print("done.")
        """

        self.model = DCNModel(BATCH_SIZE, self.device).to(self.device) 
        self.params = self.model.parameters()
        self.optimizer = optim.Adam(self.params, lr=0.1, amsgrad=True)
        start_epoch = 0

        """
        for i, p in enumerate(self.params):
            print(i, p)
        """
        
        # Continue training from a saved serialised model.
        if state_file_path is not None:
            if not os.path.isfile(state_file_path):
                print("Failed to read path %s, aborting." % state_file_path)
                return
            state = th.load(state_file_path)
            if len(state) != 3:
                print("Invalid state read from path %s, aborting. State keys: %s" % (state_file_path, state.keys()))
                return
            start_epoch = state[SERIALISATION_KEY_EPOCH] + 1
            self.model.load_state_dict(state[SERIALISATION_KEY_MODEL])
            self.optim.load_state_dict(state[SERIALISATION_KEY_OPTIM])

        self.emb_mat, self.word2id, self.id2word = get_glove(self.glove_path, EMBEDDING_DIM) 

        curr_dir_path = str(pathlib.Path().absolute())
        serial_path = "/model/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(curr_dir_path+serial_path)
        
        for epoch in range(start_epoch, NUM_EPOCHS):
            print("-" * 50)
            print("Training Epoch %i" % epoch)
            epoch_tic = time.time()
                         
            for batch in get_batch_generator(self.word2id, self.context_path, self.question_path, self.ans_path, 64, context_len=MAX_CONTEXT_LEN,
                    question_len=MAX_QUESTION_LEN, discard_long=True):
                
                print("Training global step %i" % self.global_step)
                self.global_step += 1
                
                # TODO build doc and que matrix
                loss, param_norm, grad_norm = self.train_one_batch(batch, self.model, self.optimizer, self.params)

            epoch_toc = time.time()
            epoch_time = epoch_toc - epoch_tic
            print("Epoch %i completed in %i seconds" % (epoch, epoch_time))

            # save model after each epoch:
            print("Saving training state ... ", end='')
            state = {SERIALISATION_KEY_EPOCH: epoch, SERIALISATION_KEY_MODEL: self.model.state_dict(), SERIALISATION_KEY_OPTIM: self.optimizer.state_dict() }
            th.save(state, serial_path + "epoch_%i.par" % epoch) 
            print("done.")

Training().training()
