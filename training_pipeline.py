# -*- coding: utf-8 -*-
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
import math
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


# the idea is that this will stand out on the loss graph
def filter_nan(x):
    return -1000 if math.isnan(x) else x

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


def save_state(serial_path, next_batch, next_epoch, next_global_step, model, optim):
    target_filename = serial_path + "epoch%d_batch%d.par" % (next_epoch,next_batch)
    print("Saving training state to '%s'... " % target_filename, end='')
    state = {
        SERIALISATION_KEY_BATCH: next_batch, 
        SERIALISATION_KEY_EPOCH: next_epoch, 
        SERIALISATION_KEY_GLOBAL_STEP: next_global_step,
        SERIALISATION_KEY_MODEL: model.state_dict(), 
        SERIALISATION_KEY_OPTIM: optim.state_dict() }
    th.save(state, target_filename) 
    print("done.")


class Training:


    def useEntireTrainingSet(self):
        self.question_path = "preprocessing/data/preprocessed_train_question.txt"
        self.context_path = "preprocessing/data/preprocessed_train_context.txt"
        self.ans_path = "preprocessing/data/preprocessed_train_ans_span.txt"

    def useTrainingSubset1(self):
        # Train on just the first document within the training set
        self.question_path = "preprocessing/data/subset-1/preprocessed_train-subset-1_question.txt"
        self.context_path = "preprocessing/data/subset-1/preprocessed_train-subset-1_context.txt"
        self.ans_path = "preprocessing/data/subset-1/preprocessed_train-subset-1_ans_span.txt"
        
    def useTrainingSubset2(self):
    	# Train on just the first paragraph of the first document in training set
    	# (15 questions)
    	self.question_path = "preprocessing/data/subset-2/preprocessed_train-subset-2_question.txt"
        self.context_path = "preprocessing/data/subset-2/preprocessed_train-subset-2_context.txt"
        self.ans_path = "preprocessing/data/subset-2/preprocessed_train-subset-2_ans_span.txt"

    def checkTrainingPaths(self):
        if self.question_path is None or self.context_path is None or self.ans_path is None:
            print("The question/context/context paths have not been set...aborting.")
            sys.exit(0)
        else:
            print("Training with:\nQuestion path:%s\nContext path:%s\nAnswer path:%s\n" % (self.question_path, self.context_path, self.context_path))

    def __init__(self):
        self.device = th.device("cuda:0" if th.cuda.is_available() and (not DISABLE_CUDA) else "cpu")
        
        self.global_step = None
        self.model = None
        self.optimizer = None
        self.params = " "  # Model parameters (50 layers, infeatures:200, outfeatures:200)

        #TO BE UPDATED  #TODO: What does this comment mean?
        self.word2id_path = "word_vectors/word2id.csv"
        self.id2word_path = "word_vectors/id2word.csv"
        self.glove_path = "word_vectors/glove.840B.300d.txt"
        self.emb_mat_path = "word_vectors/embedding_matrix.txt"
        # These get set dynamically by a call to one of the methods above
        self.question_path = None
        self.context_path = None
        self.ans_path = None

    # Convert question mask, ids; context mask, ids; answer start spans, answer end spans to tensors
    def get_data(self, batch, is_train=True):
        qn_mask = get_mask_from_seq_len(batch.qn_mask)
        qn_mask_var = th.from_numpy(qn_mask).long().to(self.device)

        context_mask = get_mask_from_seq_len(batch.context_mask)
        context_mask_var = th.from_numpy(context_mask).long().to(self.device)

        qn_seq_var = th.from_numpy(batch.qn_ids).long().to(self.device)
        context_seq_var = th.from_numpy(batch.context_ids).long().to(self.device)

        if is_train:
            span_s, span_e = self.get_spans(batch.ans_span)
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var, span_s, span_e 
        else:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var


    def get_spans(self, span):
        def to_th(x):
            return th.tensor(list(x)).long().to(self.device)
        
        span_s = to_th(map(lambda p: p[0], span))
        span_e = to_th(map(lambda p: p[1], span))
        return span_s, span_e


    def load_saved_state(self, state_file_path):
        global_step = 0
        start_batch = 0
        start_epoch = 0
        
        # Continue training from a saved serialised model.
        if state_file_path is not None:
            if not os.path.isfile(state_file_path):
                print("Failed to read path %s, aborting." % state_file_path)
                return
            state = th.load(state_file_path)
            if len(state) != 5:
                print("Invalid state read from path %s, aborting. State keys: %s" % (state_file_path, state.keys()))
                return
            global_step = state[SERIALISATION_KEY_GLOBAL_STEP]
            start_batch = state[SERIALISATION_KEY_BATCH]
            start_epoch = state[SERIALISATION_KEY_EPOCH] + 1
            self.model.load_state_dict(state[SERIALISATION_KEY_MODEL])
            self.optimizer.load_state_dict(state[SERIALISATION_KEY_OPTIM])

            print("Loaded saved state successfully, see below:")
            print("- Upcoming epoch: %d." % start_epoch)
            print("- Upcoming batch index: %d." % start_batch)
            print("- Upcoming global step: %d." % global_step)
            print("Resuming training...")

        return global_step, start_batch, start_epoch


    def seq_to_emb(self, seq):
        seq_list = seq.tolist()
        emb_list = [[self.emb_mat[y] for y in x] for x in seq_list]
        return th.tensor(emb_list, dtype=th.float32, device=self.device)


    def train_one_batch(self, batch, model, optimizer, params):
        optimizer.zero_grad()
        q_seq, q_lens, d_seq, d_lens, span_s, span_e = self.get_data(batch)
        
        # convert sequence into embedding
        q_emb = self.seq_to_emb(q_seq)  #Batched questions embedding Shape: batch_size X max_question_length, embedding_dimension
        d_emb = self.seq_to_emb(d_seq)  #Batched contexts embedding Shape:  batch_size X max_context_length, embedding_dimension

        loss, _, _ = model(d_emb, q_emb, span_s, span_e)

        l2_reg = 0.0 
        for W in params:
            l2_reg = l2_reg + W.norm(2)
        loss = loss + REG_LAMBDA * l2_reg
        
        param_norm = get_param_norm(params)
        grad_norm = get_grad_norm(params)

        clip_grad_norm_(params, MAX_GRAD_NORM)
       
        print("loss (incl. reg):", loss)
        with open("./loss.log", "a") as f:
            f.write("%i: %i\n" % (self.global_step, filter_nan(loss)))

        loss.backward()
        optimizer.step()
        
        return loss.item(), param_norm, grad_norm


    # Pass state_file_path to resume training from an existing checkpoint.
    def training(self, state_file_path=None):
        self.checkTrainingPaths()

        self.model = DCNModel(BATCH_SIZE, self.device).to(self.device).train()
        self.params = self.model.parameters()
        self.optimizer = optim.Adam(self.params, lr=0.1, amsgrad=True) # TODO: choose right hyperparameters

        # Load saved state from the path. If path is None, still do call this method!
        self.global_step, start_batch, start_epoch = self.load_saved_state(state_file_path)

        # Load glove embeddings. Takes a bit of time.
        self.emb_mat, self.word2id, self.id2word = get_glove(self.glove_path, EMBEDDING_DIM) 

        # Create directory for this training session.
        curr_dir_path = str(pathlib.Path().absolute())
        serial_path = curr_dir_path + "/model/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
        os.makedirs(serial_path)
        print("This training session will be saved at:\n%s" % serial_path)
        
        # Train / resume training.
        for epoch in range(start_epoch, NUM_EPOCHS):
            print("-" * 50)
            print("Training Epoch %i" % epoch)
            epoch_tic = time.time()
                        
            for batch_ind, batch in enumerate(get_batch_generator(
                    self.word2id, self.context_path, self.question_path, 
                    self.ans_path, 64, context_len=MAX_CONTEXT_LEN,
                    question_len=MAX_QUESTION_LEN, discard_long=True)):
                
                # Skip first start_batch batches (if resuming training from saved state).
                if start_batch != 0:
                    start_batch -= 1
                else:
                    print("About to train global step %i..." % self.global_step)
                    self.global_step += 1
                    loss, param_norm, grad_norm = self.train_one_batch(batch, self.model, self.optimizer, self.params)

                    # Save state at a configurable frequency.
                    if self.global_step % TRAINING_SAVE_FREQUENCY == 1:  # 1 so that the first save is as early as possible.
                        save_state(serial_path, batch_ind+1, epoch, self.global_step, self.model, self.optimizer)

            epoch_toc = time.time()
            epoch_time = epoch_toc - epoch_tic
            print("!*" * 50)
            print("Epoch %i completed in %i seconds" % (epoch, epoch_time))
            print("!*" * 50)
            
            # Save state at the end of epoch.
            save_state(serial_path, 0, epoch+1, self.global_step, self.model, self.optimizer)


# TODO: Move.
saved_state_path = None if len(sys.argv) <= 1 else sys.argv[1]

training_pipeline = Training()

# Specify the training set you want use here:
training_pipeline.useTrainingSubset2()

training_pipeline.training(saved_state_path)

