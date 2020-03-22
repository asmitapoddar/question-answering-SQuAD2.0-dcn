import io
import json
import logging
import os
import sys
import time
import numpy as np
import torch

# from preprocessing/batching import get_batch_generator

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
	    
        loss = loss + config.reg_lambda * l2_reg

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
            
      for batch in get_batch_generator(word2id_path, context_path, question_path, ans_path, 64, context_len=600,
                                             question_len=30, discard_long=True):
          global_step += 1
          loss, param_norm, grad_norm = self.train_one_batch(batch, model, optimizer, params)
      iter_toc = time.time()
      iter_time = iter_toc - iter_tic

  loss, param_norm, grad_norm = self.train_one_batch(batch, model, optimizer, params)
   iter_toc = time.time()
   iter_time = iter_toc - iter_tic
