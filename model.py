# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline


# -------------- TODO -------------- #
# 1. Finish batchifying all code:
#    - fix DynamicPointerDecoder's forward pass.
#    - Go through all tensor operations and LSTMs, etc. until everything works.
# 2. Verify that GPU is used as we're expecting.


import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time

th.manual_seed(1)

# Test flags.
TEST_DCN_MODEL = True
TEST_DYNAMIC_POINTER_DECODER = False
TEST_DYNAMIC_POINTER_DECODER_2 = False
TEST_HMN = False

# Defaults/constants.
BATCH_SIZE = 64
DROPOUT = 0.3
HIDDEN_DIM = 200  # Denoted by 'l' in the paper.
MAXOUT_POOL_SIZE = 16

# The encoder LSTM.
class Encoder(nn.Module):
  def __init__(self, doc_word_vecs, que_word_vecs, hidden_dim, batch_size, dropout, device):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.hidden_dim = hidden_dim
    self.device = device

    # Dimensionality of word vectors.
    self.word_vec_dim = doc_word_vecs.size()[2]
    assert(self.word_vec_dim == que_word_vecs.size()[2])

    # Dimension of the hidden state and cell state (they're equal) of the LSTM
    self.lstm = nn.LSTM(self.word_vec_dim, hidden_dim, 1, batch_first=True, bidirectional=False, dropout=dropout)

  def generate_initial_hidden_state(self):
    # Even if batch_first=True, the initial hidden state should still have batch index in dim1, not dim0.
    return (th.zeros(1, self.batch_size, self.hidden_dim, device=self.device),
            th.zeros(1, self.batch_size, self.hidden_dim, device=self.device))

  def forward(self, x, hidden):
    return self.lstm(x, hidden)


# Takes in D, Q. Produces U.
class CoattentionModule(nn.Module):
    def __init__(self, batch_size, dropout, hidden_dim, device):
        super(CoattentionModule, self).__init__()
        self.batch_size = batch_size
        self.bilstm_encoder = BiLSTMEncoder(hidden_dim, batch_size, dropout, device)
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, D_T, Q_T):
        #Q: B x n + 1 x l
        #D: B x m + 1 x l
        
        Q = th.transpose(Q_T, 1, 2) #B x  n + 1 x l
        D = th.transpose(D_T, 1, 2) #B x m + 1 x l

        # Coattention.
        print("coattention_module D: ", D.size())
        print("coattention_module Q: ", Q.size())

        L = th.bmm(D_T, Q) # L = B x m + 1 x n + 1
        AQ = F.softmax(L, dim=1) # B x(m+1)×(n+1)
        AD_T = F.softmax(L,dim=2) # B x(m+1)×(n+1)
        AD = th.transpose(AD_T, 1, 2) # B x (n + 1) x (m + 1)

        CQ = th.bmm(D,AQ) # l×(n+1)
        CD = th.bmm(th.cat((Q,CQ),1),AD) # B x 2l x m + 1
        C_D_t = th.transpose(CD, 1, 2)  # B x m + 1 x 2l

        # Fusion BiLSTM.
        input_to_BiLSTM = th.transpose(th.cat((D,CD), dim=1), 1, 2) # B x (m+1) x 3l
        print("input_to_BiLSTM.size():",input_to_BiLSTM.size())

        U = self.bilstm_encoder(input_to_BiLSTM)
        return U


class BiLSTMEncoder(nn.Module):
    def __init__(self, hidden_dim, batch_size, dropout, device):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.dropout = dropout
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=dropout)

    def init_hidden(self):
        # TODO: Is initialisation zeros or randn? 
        # First is the hidden h, second is the cell c.
        return (th.zeros(2, self.batch_size, self.hidden_dim, device=self.device),
              th.zeros(2, self.batch_size,self.hidden_dim, device=self.device))

    def forward(self, input_BiLSTM):
        lstm_out, self.hidden = self.lstm(
            input_BiLSTM.reshape(input_BiLSTM.shape[0], input_BiLSTM.shape[1], -1), 
            self.hidden)
        return lstm_out


class HighwayMaxoutNetwork(nn.Module):
  def __init__(self, batch_size, dropout, hidden_dim, maxout_pool_size, device): 
    super(HighwayMaxoutNetwork, self).__init__()

    self.hidden_dim = hidden_dim
    self.maxout_pool_size = MAXOUT_POOL_SIZE
    self.device = device

    # Don't apply dropout to biases.
    self.dropout = nn.Dropout(p=DROPOUT)

    # W_D := Weights of MLP applied to the coattention encodings of
    # the start/end positions, and the current LSTM hidden state (h_i)
    # (nn.Linear is an affine transformation y = Wx + b).

    # There are 5 * hidden_dim incoming features (u_si-1, u_ei-1, h_i) 
    # which are vectors containing (2l, 2l, l) elements respectively.
    # There's l outgoing features (i.e. r).
    # There's no bias for this MLP.
    
    # (From OpenReview) random initialisation is used for W's and b's
    self.W_D = self.dropout(nn.Parameter(th.randn(self.hidden_dim, 5 * self.hidden_dim)))

    # 1st Maxout layer
    self.W_1 = self.dropout(nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim, 3 * self.hidden_dim)))
    self.b_1 = nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim))

    # 2nd maxout layer
    self.W_2 = self.dropout(nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim, self.hidden_dim)))
    self.b_2 = nn.Parameter(th.randn(self.maxout_pool_size, self.hidden_dim))

    # 3rd maxout layer
    self.W_3 = self.dropout(nn.Parameter(th.randn(self.maxout_pool_size, 1, 2 * self.hidden_dim)))
    self.b_3 = nn.Parameter(th.randn(self.maxout_pool_size, 1))


  def forward(self, u_t, h_i, u_si_m_1, u_ei_m_1):

    assert(u_t.size()[0] == 2 * self.hidden_dim) 
    assert(u_t.size()[1] == 1) 
    assert(h_i.size()[0] == self.hidden_dim)
    assert(h_i.size()[1] == 1)
    assert(u_si_m_1.size()[0] == 2 * self.hidden_dim)
    assert(u_si_m_1.size()[1] == 1)
    assert(u_ei_m_1.size()[0] == 2 * self.hidden_dim)
    assert(u_ei_m_1.size()[1] == 1)

    # r := output of MLP.
    r = th.tanh(self.W_D.mm(th.cat((h_i, u_si_m_1, u_ei_m_1), 0)))

    # m_t_1 := output of 1st maxout layer.
    m_t_1_beforemaxpool = th.mm(
        self.W_1.view(self.maxout_pool_size * self.hidden_dim, 
                        3 * self.hidden_dim), 
                        th.cat((u_t, r), 0)
    ).view(
        self.maxout_pool_size, 
        self.hidden_dim, 
        1
    ).squeeze() + self.b_1

    # The max operation in Eq.9-12 computes the maximum value over the 
    # first dimension of a tensor.
    m_t_1 = th.Tensor.max(m_t_1_beforemaxpool, dim=0).values.unsqueeze(dim=1)

    m_t_2_beforemaxpool = th.mm(
        self.W_2.view(self.maxout_pool_size * self.hidden_dim, 
                        self.hidden_dim), 
                        m_t_1
    ).view(
        self.maxout_pool_size, 
        self.hidden_dim, 
        1
    ).squeeze() + self.b_2
    m_t_2 = th.Tensor.max(m_t_2_beforemaxpool, dim=0).values.unsqueeze(dim=1)

    # HMN output
    output_beforemaxpool = th.mm(
        self.W_3.view(
            self.maxout_pool_size, 
            2 * self.hidden_dim
        ), 
        # highway connection
        th.cat((m_t_1, m_t_2), 0)
    ) + self.b_3
    
    output = th.Tensor.max(output_beforemaxpool, dim=0).values
    return output

class DynamicPointerDecoder(nn.Module):
  def __init__(self, batch_size, dropout_hmn, dropout_lstm, hidden_dim, device):
    super(DynamicPointerDecoder, self).__init__()
    self.device = device
    self.hmn_alpha = HighwayMaxoutNetwork(batch_size, dropout_hmn, hidden_dim, MAXOUT_POOL_SIZE, device)
    self.hmn_beta = HighwayMaxoutNetwork(batch_size, dropout_hmn, hidden_dim, MAXOUT_POOL_SIZE, device)
    self.lstm = nn.LSTM(4*hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False, dropout=dropout_lstm)

  def forward(self, U, max_iter):

    # TODO: Value to choose for max_iter (600?)
    # Initialise h_0, s_i_0, e_i_0 (TODO can change)
    s = 0
    e = 0
    
    # initialize the hidden and cell states 
    # hidden = (h, c)
    doc_length = U.size()[1]
    hidden = (th.randn(1,1,HIDDEN_DIM), th.randn(1,1,HIDDEN_DIM))
    
    # "The iterative procedure halts when both the estimate of the start position 
    # and the estimate of the end position no longer change, 
    # or when a maximum number of iterations is reached"

    # We build up the losses here (the iteration being the first dimension)
    alphas = th.tensor([], device=self.device).view(0, doc_length)
    betas = th.tensor([], device=self.device).view(0, doc_length)

    # TODO: make it run only until convergence (or maxiter)
    for _ in range(max_iter):
      # call LSTM to update h_i

      # Step through the sequence one element at a time.
      # after each step, hidden contains the hidden state.
      u_concatenated = th.cat((U[:,s], U[:,e]), dim=0).view(-1, 1)
      #print("u_concatenated.size()", u_concatenated.size())

      out, hidden = self.lstm(u_concatenated.view(1, 1, -1), hidden)
      h, _ = hidden

      # Call HMN to update s_i, e_i
      alpha = th.tensor([], device=self.device).view(0, 1)
      beta = th.tensor([], device=self.device).view(0, 1)

      for t in range(doc_length):
        t_hmn_alpha = self.hmn_alpha(U[:, t].view(-1, 1), h.view(-1, 1), U[:,s].view(-1, 1), U[:,e].view(-1, 1))
        t_hmn_beta = self.hmn_beta(U[:, t].view(-1, 1), h.view(-1, 1), U[:,s].view(-1, 1), U[:,e].view(-1, 1))
        alpha = th.cat((alpha, t_hmn_alpha.view(1, 1)), dim=0)
        beta = th.cat((beta, t_hmn_beta.view(1, 1)), dim=0)

      # Leaving out dim=0 changes behaviour
      _, s = th.max(alpha, dim=0)
      _, e = th.max(beta, dim=0)
      
      alphas = th.cat((alphas, alpha.view(1, -1)), dim=0)
      betas = th.cat((betas, beta.view(1, -1)), dim=0)

    return (alphas, betas, s, e)

# The full model.
class DCNModel(nn.Module):
  def __init__(
      self, doc_word_vecs, que_word_vecs, batch_size, device, hidden_dim=HIDDEN_DIM, dropout_encoder=DROPOUT, 
      dropout_coattention=DROPOUT, dropout_decoder_hmn=DROPOUT, dropout_decoder_lstm=DROPOUT):
    super(DCNModel, self).__init__()
    self.batch_size = batch_size
    self.coattention_module = CoattentionModule(batch_size, dropout_coattention, hidden_dim, device)
    self.dyn_ptr_dec = DynamicPointerDecoder(batch_size, dropout_decoder_hmn, dropout_decoder_lstm, hidden_dim, device) 
    self.encoder = Encoder(doc_word_vecs, que_word_vecs, hidden_dim, batch_size, dropout_encoder, device)
    self.encoder_sentinel = nn.Parameter(th.randn(batch_size, 1, hidden_dim)) # the sentinel is a trainable parameter of the network
    self.hidden_dim = hidden_dim
    self.WQ = nn.Linear(hidden_dim, hidden_dim)


  def forward(self, doc_word_vecs, que_word_vecs):
    # doc_word_vecs should have 3 dimensions: [batch_size, num_docs_in_batch, word_vec,dim].
    # que_word_vecs the same as above.

    # TODO: how should we initialise the hidden state of the LSTM? For now:
    initial_hidden = self.encoder.generate_initial_hidden_state()
    outp, _ = self.encoder(doc_word_vecs, initial_hidden)
    # outp: B x m x l
    D_T = th.cat([outp, self.encoder_sentinel], dim=1)  # append sentinel word vector # l X n+1
    # D: B x (m+1) x l

    # TODO: Make sure we should indeed reinit hidden state before encoding the q.
    outp, _ = self.encoder(que_word_vecs, initial_hidden)
    Qprime = th.cat([outp, self.encoder_sentinel], dim=1)  # append sentinel word vector
    # Qprime: B x (n+1) x l
    Q_T = th.tanh(self.WQ(Qprime.view(-1, self.hidden_dim))).view(Qprime.size())
    # Q: B x (n+1) x l

    U = self.coattention_module(D_T,Q_T)
    
    # TODO: Replace with the true start/end positions for the current batch of questions
    x,y = th.randint(0, doc_word_vecs.size()[0], (2,))
    x, y = min(x,y), max(x,y)
    
    # TODO: Replace with actual U
    max_iters = 10
    doc_words = doc_word_vecs.size()[0]

    criterion = nn.CrossEntropyLoss()    

    # Run the dynamic pointer decoder, accumulate the 
    # distributions over the start positions (alphas)
    # and end positions (betas) at each iteration
    # as well as the final start/end indices
    alphas, betas, start, end = self.dyn_ptr_dec.forward(U, max_iters)

    # Accumulator for the losses incurred across 
    # iterations of the dynamic pointing decoder
    loss = th.FloatTensor([0])
    for it in range(max_iters):
      loss += criterion(alphas[it].view(1, -1), Variable(th.LongTensor([x])))
      loss += criterion(betas[it].view(1, -1), Variable(th.LongTensor([x])))
 
    return loss, start, end


# Optimiser.
def run_optimiser():
    doc = th.randn(64, 30, 200) # Fake word vec dimension set to 200.
    que = th.randn(64, 5, 200)  # Fake word vec dimension set to 200.
    model = DCNModel(doc, que, BATCH_SIZE)

    # TODO: hyperparameters?
    optimizer = optim.Adam(model.parameters())
    n_iters = 1000

    # TODO: batching?
    for iter in range(n_iters):
        optimizer.zero_grad()
        loss, _, _ = model(doc, que)
        loss.backward()
        optimizer.step()


# ------------------------------- tests -------------------------------#

# DCNModel Test.
def test_dcn_model():
    # Is GPU available:
    print ("cuda device count = %d" % th.cuda.device_count())
    print ("cuda is available = %d" % th.cuda.is_available())
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    doc = th.randn(64, 30, 200, device=device) # Fake word vec dimension set to 200.
    que = th.randn(64, 5, 200, device=device)  # Fake word vec dimension set to 200.

    # Run model.
    model = DCNModel(doc, que, BATCH_SIZE, device)
    if th.cuda.is_available:
      # https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/2
      # Move net to gpu:
      model = model.cuda()
    loss, s, e = model.forward(doc, que)
    print("loss: ", loss, ", s:", s, ", e:", e)
    model.zero_grad()
    loss.backward()

    print("%d/%d parameters have non-None gradients." % (len([param for param in model.parameters() if param.grad is not None]), len(list(model.parameters()))))

if TEST_DCN_MODEL:
    test_dcn_model()

def test_hmn():
    hmn = HighwayMaxoutNetwork(BATCH_SIZE, HIDDEN_DIM, MAXOUT_POOL_SIZE) #TODO device
    u_t = th.ones(2 * HIDDEN_DIM, 1) #TODO device
    h_i = th.ones(HIDDEN_DIM, 1) #TODO device
    u_si_m_1, u_ei_m_1 = th.ones(2 * HIDDEN_DIM, 1), th.ones(2 * HIDDEN_DIM, 1) #TODO device
    output = hmn.forward(u_t, h_i, u_si_m_1, u_ei_m_1)
    #print(output)
    output.backward()

if TEST_HMN:
    test_hmn()

def test_decoder():
    dpe = DynamicPointerDecoder()
    alphas, betas, _, _ = dpe.forward(th.randn(2 * HIDDEN_DIM, 50), 10) #TODO device
    dpe.zero_grad()

if TEST_DYNAMIC_POINTER_DECODER:
    test_decoder()

def test_decoder_2():
    dpd = DynamicPointerDecoder(BATCH_SIZE, DROPOUT, HIDDEN_DIM) #TODO device
    max_iter = 10
    U = th.ones(2 * HIDDEN_DIM, 50) #TODO device
    alphas, betas, s, e = dpd.forward(U, max_iter)
    loss = th.mean(th.mean(alphas, dim=0)) + th.mean(th.mean(betas, dim=0))
    loss.backward()
    print(loss)

if TEST_DYNAMIC_POINTER_DECODER_2:
    test_decoder_2()
