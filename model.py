from constants import *
from index_convergence import *

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time

th.manual_seed(RANDOM_SEED)

# The encoder LSTM.
class Encoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, batch_size, dropout, device):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.device = device

    # Dimension of the hidden state and cell state (they're equal) of the LSTM
    self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 1, batch_first=True, bidirectional=False)

  def generate_initial_hidden_state(self):
    # Even if batch_first=True, the initial hidden state should still have batch index in dim1, not dim0.
    return (th.zeros(1, self.batch_size, self.hidden_dim, device=self.device, dtype=th.float32),
            th.zeros(1, self.batch_size, self.hidden_dim, device=self.device, dtype=th.float32))
    
    #return nn.init.xavier_initialisation(th.zeros(1, self.batch_size, self.hidden_dim, device=self.device, dtype=th.float32),
    #      th.zeros(1, self.batch_size,self.hidden_dim, device=self.device, dtype=th.float32))

  def forward(self, x):
    hidden = self.generate_initial_hidden_state()
    lstm_out, hidden = self.lstm(x, hidden)
    return lstm_out


# Takes in D, Q. Produces U.
class CoattentionModule(nn.Module):
    def __init__(self, batch_size, dropout, embedding_dim, hidden_dim, device):
        super(CoattentionModule, self).__init__()
        self.batch_size = batch_size
        self.bilstm_encoder = BiLSTMEncoder(embedding_dim, hidden_dim, batch_size, dropout, device)
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, D_T, Q_T):
        #Q: B x n + 1 x l
        #D: B x m + 1 x l
        
        Q = th.transpose(Q_T, 1, 2) #B x  n + 1 x l
        D = th.transpose(D_T, 1, 2) #B x m + 1 x l

        # Coattention.
        L = th.bmm(D_T, Q) # L = B x m + 1 x n + 1
        AQ = F.softmax(L, dim=1) # B x(m+1)×(n+1)
        AD_T = F.softmax(L,dim=2) # B x(m+1)×(n+1)
        AD = th.transpose(AD_T, 1, 2) # B x (n + 1) x (m + 1)

        CQ = th.bmm(D,AQ) # l×(n+1)
        CD = th.bmm(th.cat((Q,CQ),1),AD) # B x 2l x m + 1
        C_D_t = th.transpose(CD, 1, 2)  # B x m + 1 x 2l

        # Fusion BiLSTM.
        input_to_BiLSTM = th.transpose(th.cat((D,CD), dim=1), 1, 2) # B x (m+1) x 3l

        U = self.bilstm_encoder(input_to_BiLSTM)
        return U


class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, dropout, device):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.dropout = dropout
        self.lstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)

    def init_hidden(self): 
        # First is the hidden h, second is the cell c.
        return (th.zeros(2, self.batch_size, self.hidden_dim, device=self.device, dtype=th.float32),
              th.zeros(2, self.batch_size,self.hidden_dim, device=self.device, dtype=th.float32))
        # return nn.init.xavier_initialisation(th.zeros(2, self.batch_size, self.hidden_dim, device=self.device, dtype=th.float32),
        #      th.zeros(2, self.batch_size,self.hidden_dim, device=self.device, dtype=th.float32))

    def forward(self, input_BiLSTM):
        hidden = self.init_hidden()
        lstm_out, hidden = self.lstm(input_BiLSTM, hidden)
        U = th.transpose(lstm_out, 1, 2)[:,:,1:]
        return U


class HighwayMaxoutNetwork(nn.Module):
  def __init__(self, batch_size, dropout, hidden_dim, maxout_pool_size, device): 
    super(HighwayMaxoutNetwork, self).__init__()

    self.batch_size = batch_size
    self.device = device
    self.dropout = dropout
    self.hidden_dim = hidden_dim
    self.maxout_pool_size = maxout_pool_size

    self.dropout_r = nn.Identity() if DISABLE_HMN_DROPOUT else nn.Dropout(p=dropout)
    self.dropout_mt1 = nn.Identity() if DISABLE_HMN_DROPOUT else nn.Dropout(p=dropout)
    self.dropout_mt2 = nn.Identity() if DISABLE_HMN_DROPOUT else nn.Dropout(p=dropout)

    # W_D := Weights of MLP applied to the coattention encodings of
    # the start/end positions, and the current LSTM hidden state (h_i)
    # (nn.Linear is an affine transformation y = Wx + b).

    # There are 5 * hidden_dim incoming features (u_si-1, u_ei-1, h_i) 
    # which are vectors containing (2l, 2l, l) elements respectively.
    # There's l outgoing features (i.e. r).
    # There's no bias for this MLP.
    
    # (From OpenReview) random initialisation is used for W's and b's
    self.W_D = nn.Parameter(th.randn(self.hidden_dim, 5 * self.hidden_dim, device=device))

    # 1st Maxout layer
    self.W_1 = nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim, 3 * self.hidden_dim, device=device))
    self.b_1 = nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim, device=device))

    # 2nd maxout layer
    self.W_2 = nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim, self.hidden_dim, device=device))
    self.b_2 = nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim, device=device))

    # 3rd maxout layer
    self.W_3 = nn.Parameter(th.randn(self.maxout_pool_size, 1, 2 * self.hidden_dim, device=device))
    self.b_3 = nn.Parameter(th.randn(self.maxout_pool_size, 1, device=device))

  def forward(self, u_t, h_i, u_si_m_1, u_ei_m_1):

    assert(u_t.size()[0+1] == 2 * self.hidden_dim)
    assert(u_t.size()[1+1] == 1)
    assert(h_i.size()[0+1] == self.hidden_dim)
    assert(h_i.size()[1+1] == 1)
    assert(u_si_m_1.size()[0+1] == 2 * self.hidden_dim)
    assert(u_si_m_1.size()[1+1] == 1)
    assert(u_ei_m_1.size()[0+1] == 2 * self.hidden_dim)
    assert(u_ei_m_1.size()[1+1] == 1)

    # MULTILAYER PERCEPTRON (r)

    # Concatenate current LSTM state with coattention encodings of
    # current estimates for start and end positions of answer span.
    h_us_ue = th.cat((h_i, u_si_m_1, u_ei_m_1), dim=1)
    assert(h_us_ue.size()[0] == self.batch_size)
    assert(h_us_ue.size()[1+0] == 5 * self.hidden_dim)
    assert(h_us_ue.size()[1+1] == 1)

    # r := output of MLP
    r = th.tanh(self.W_D.matmul(h_us_ue))
    r = self.dropout_r(r)

    # r has dimension BATCH_SIZE * HIDDEN_DIM * 1
    assert(r.size()[0] == self.batch_size)
    assert(r.size()[1+0] == self.hidden_dim)
    assert(r.size()[1+1] == 1)

    # m_t_1 := output of 1st maxout layer (Eq. 11 in the paper)
    w1_reshaped = self.W_1.view(self.maxout_pool_size * self.hidden_dim, 3 * self.hidden_dim)

    u_r = th.cat((u_t, r), dim=1).squeeze(dim=2).transpose(0, 1)
    # Note that the batch dimension here isn't the first one.
    assert(u_r.size()[0] == 3 * self.hidden_dim)
    assert(u_r.size()[1] == self.batch_size)

    # Transpose the result of matmul(w1_reshaped, u_r) so that BATCH_SIZE is again the first dimension
    m_t_1_beforemaxpool = th.mm(
        w1_reshaped,
        u_r).transpose(0, 1).view(self.batch_size, self.maxout_pool_size, self.hidden_dim) + self.b_1.expand(self.batch_size, -1, -1)
    assert(m_t_1_beforemaxpool.size()[0] == self.batch_size)
    assert(m_t_1_beforemaxpool.size()[1+0] == self.maxout_pool_size)
    assert(m_t_1_beforemaxpool.size()[1+1] == self.hidden_dim)

    m_t_1 = th.Tensor.max(m_t_1_beforemaxpool, dim=1).values
    m_t_1 = self.dropout_mt1(m_t_1)
    assert(m_t_1.size()[0] == self.batch_size)
    assert(m_t_1.size()[1+0] == self.hidden_dim)

    # Eq. 12 in the paper
    m_t_2_beforemaxpool = th.mm(
        self.W_2.view(self.maxout_pool_size * self.hidden_dim, self.hidden_dim),
        m_t_1.transpose(0, 1)
    ).transpose(0, 1).view(
        self.batch_size,
        self.maxout_pool_size, 
        self.hidden_dim
    ) + self.b_2.expand(self.batch_size, -1, -1)
    m_t_2 = th.Tensor.max(m_t_2_beforemaxpool, dim=1).values
    m_t_2 = self.dropout_mt2(m_t_2)
    assert(m_t_2.size()[0] == self.batch_size)
    assert(m_t_2.size()[1+0] == self.hidden_dim)

    # HMN output (Eq. 9 in the paper)
    output_beforemaxpool = th.mm(
        self.W_3.view(
            self.maxout_pool_size * 1,
            2 * self.hidden_dim
        ), 
        # highway connection
        th.cat((m_t_1, m_t_2), 1).transpose(0, 1)
    ).transpose(0, 1).view(self.batch_size, self.maxout_pool_size, 1) + self.b_3.expand(self.batch_size, -1, -1)
    
    output = th.Tensor.max(output_beforemaxpool, dim=1).values
    assert(output.size()[0] == self.batch_size)
    assert(output.size()[1] == 1)

    return output

class DynamicPointerDecoder(nn.Module):
  def __init__(self, batch_size, max_iter, dropout_hmn, dropout_lstm, hidden_dim, maxout_pool_size, device):
    super(DynamicPointerDecoder, self).__init__()
    self.batch_size = batch_size
    self.convergence = None
    self.device = device
    self.hidden_dim = hidden_dim
    self.hmn_alpha = HighwayMaxoutNetwork(batch_size, dropout_hmn, hidden_dim, maxout_pool_size, device)
    self.hmn_beta = HighwayMaxoutNetwork(batch_size, dropout_hmn, hidden_dim, maxout_pool_size, device)
    self.lstm = nn.LSTM(4*hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
    self.max_iter = max_iter

  def forward(self, U):

    assert(U.size()[0] == self.batch_size)

    # TODO: Value to choose for max_iter (600?)
    # Initialise h_0, s_i_0, e_i_0 (TODO can change)
    s = th.zeros(self.batch_size, device=self.device, dtype=th.long)
    e = th.zeros(self.batch_size, device=self.device, dtype=th.long)
    
    # initialize the hidden and cell states 
    # hidden = (h, c)
    doc_length = U.size()[2]
    hidden = (th.zeros((1,self.batch_size,self.hidden_dim),device=self.device),
              th.zeros((1,self.batch_size,self.hidden_dim),device=self.device))
    
    # "The iterative procedure halts when both the estimate of the start position 
    # and the estimate of the end position no longer change, 
    # or when a maximum number of iterations is reached"

    # We build up the losses here (the iteration being the first dimension)
    alphas = th.tensor([], device=self.device).view(self.batch_size, 0, doc_length)
    betas = th.tensor([], device=self.device).view(self.batch_size, 0, doc_length)

    index_convergence = None

    # TODO: make it run only until convergence?
    for decoder_iter in range(self.max_iter):
      # call LSTM to update h_i

      # Step through the sequence one element at a time.
      # after each step, hidden contains the hidden state.
      s_index = s.view(-1,1,1).repeat(1,U.size()[1],1)
      u_si_m_1 = th.gather(U,2,s_index)
      e_index = e.view(-1,1,1).repeat(1,U.size()[1],1)
      u_ei_m_1 = th.gather(U,2,e_index)
      
      lstm_input = th.cat((u_si_m_1, u_ei_m_1), dim=1).view(U.size()[0], -1, 1)

      _, hidden = self.lstm(lstm_input.view(self.batch_size, 1, -1), hidden)
      h_i, _ = hidden
      h_i = h_i.squeeze(dim=0).unsqueeze(dim=2)

      # Call HMN to update s_i, e_i
      alpha = th.tensor([], device=self.device).view(self.batch_size, 0)
      beta = th.tensor([], device=self.device).view(self.batch_size, 0)

      for t in range(doc_length):
        u_t = U[:,:,t].unsqueeze(dim=2)
        t_hmn_alpha = self.hmn_alpha(u_t, h_i, u_si_m_1, u_ei_m_1)
        alpha = th.cat((alpha, t_hmn_alpha), dim=1)

      _, s = th.max(alpha, dim=1)
      
      # we want to get the effect below, using th.gather:
      # https://pytorch.org/docs/stable/torch.html#torch.gather
      # u_si = [U[batch_ind,:,s[batch_ind]] for batch_ind in range(self.batch_size)].unsqueeze(dim=2)
      # u_si = th.gather(U, ??, ??)
      s_index = s.view(-1,1,1).repeat(1,U.size()[1],1)
      u_si = th.gather(U,2,s_index)
            
      for t in range(doc_length):
        u_t = U[:,:,t].unsqueeze(dim=2)
        t_hmn_beta = self.hmn_beta(u_t, h_i, u_si, u_ei_m_1)
        beta = th.cat((beta, t_hmn_beta), dim=1)

      _, e = th.max(beta, dim=1)
      alphas = th.cat((alphas, alpha.view(self.batch_size,1,doc_length)), dim=1)
      betas = th.cat((betas, beta.view(self.batch_size,1,doc_length)), dim=1)

      index_convergence = index_convergence_update(index_convergence, s, e)

      if PRINT_SPANS_DURING_TRAINING:
        print("--- (DynamicPointerDecoder.forward) --- ")
        print("----- s (DynamicPointerDecoder.forward) iter %d ---" % decoder_iter)
        print(s)
        print("----- e (DynamicPointerDecoder.forward) iter %d ---" % decoder_iter)
        print(e)
        print("---------------------------------------")


    self.convergence = compute_index_convergence(index_convergence, self.batch_size)
    return (alphas, betas, s, e)


# The full model.
class DCNModel(nn.Module):
  def __init__(
      self, batch_size, device, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout_encoder=DROPOUT, 
      dropout_coattention=DROPOUT, dropout_decoder_hmn=DROPOUT, dropout_decoder_lstm=DROPOUT, dpd_max_iter=MAX_ITER,
      maxout_pool_size=MAXOUT_POOL_SIZE):
    super(DCNModel, self).__init__()
    self.batch_size = batch_size
    self.coattention_module = CoattentionModule(batch_size, dropout_coattention, embedding_dim, hidden_dim, device)
    self.decoder = DynamicPointerDecoder(batch_size, dpd_max_iter, dropout_decoder_hmn, dropout_decoder_lstm, hidden_dim, maxout_pool_size, device) 
    self.device = device
    self.dropout_doc_encoding = nn.Dropout(p=dropout_encoder)
    self.dropout_que_encoding = nn.Dropout(p=dropout_encoder)
    self.encoder = Encoder(embedding_dim, hidden_dim, batch_size, dropout_encoder, device)
    self.encoder_sentinel = nn.Parameter(th.randn(1, hidden_dim)) # the sentinel is a trainable parameter of the network
    self.encoder_sentinel_2 = nn.Parameter(th.randn(1, hidden_dim)) if (not DISABLE_MULTIPLE_SENTINELS) else None
    self.hidden_dim = hidden_dim
    self.WQ = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, doc_word_vecs, que_word_vecs, true_s, true_e):
    """
    doc_word_vecs: should have 3 dimensions: [batch_size, max_doc_length, word_vec_dim].
    que_word_vecs: the same as above.
    true_s: true start index of size: batch_size
    true_e: true end index of size: batch_size
    B: batch size
    m: max_doc_length
    n: max_question_length
    l: hidden_dim
    """
    
    outp = self.encoder(doc_word_vecs.float())  # outp: B x m x l
    
    D_T = th.cat([outp, self.encoder_sentinel.expand(self.batch_size, -1, -1)], dim=1)  # append sentinel word vector # l X (n+1)
    # D: B x (m+1) x l
    
    outp = self.encoder(que_word_vecs)
    Qprime = th.cat([outp, (self.encoder_sentinel_2 if not DISABLE_MULTIPLE_SENTINELS else self.encoder_sentinel).expand(self.batch_size, -1, -1)], dim=1)  # append sentinel word vector
    # Qprime: B x (n+1) x l
    Q_T = th.tanh(self.WQ(Qprime.view(-1, self.hidden_dim))).view(Qprime.size())
    # Q: B x (n+1) x l

    if not DISABLE_POST_ENCODER_DROPOUT:
      D_T = self.dropout_doc_encoding(D_T)
      Q_T = self.dropout_que_encoding(Q_T)

    U = self.coattention_module(D_T,Q_T)  # B X 2l X (m+1)
    alphas, betas, start, end = self.decoder(U)
    
    criterion = nn.CrossEntropyLoss()    

    # Accumulator for the losses incurred across 
    # iterations of the dynamic pointing decoder
    loss = th.FloatTensor([0.0]).to(self.device)

    for b in range(self.batch_size):
      for it in range(self.decoder.convergence[b] if self.decoder.convergence[b] is not None else self.decoder.max_iter):
        loss += criterion(alphas[b,it,:].view(1,-1), true_s[b].view(1))
        loss += criterion(betas[b,it,:].view(1,-1), true_e[b].view(1))
    self.decoder.convergence = None

    print("--- true_s (DCNModel.forward) ---")
    print(true_s)
    print("--- true_e (DCNModel.forward) --- ")
    print(true_e)
    return loss, start, end
