# Defaults/constants.

BATCH_SIZE = 64
DISABLE_CUDA = False # If this is set to True, make sure it should be!
DISABLE_HMN_DROPOUT = True # Investigate how to get dropout working there, if possible.
                           # Note that existing impl. has HMN dropout commented out!
                           # github.com/atulkum/co-attention/blob/master/code/model.py#L177
DISABLE_L2_REG = True
DISABLE_GRAD_CLIPPING = True
DROPOUT = 0.3
EMBEDDING_DIM = 300
HIDDEN_DIM = 200  # Denoted by 'l' in the paper.
MAX_ITER = 4
MAXOUT_POOL_SIZE = 16
MAX_CONTEXT_LEN = 600
MAX_GRAD_NORM = 0.5
MAX_QUESTION_LEN = 30
NUM_EPOCHS = 1000 
RANDOM_SEED = 1
REG_LAMBDA = 0.1
TRAINING_SAVE_FREQUENCY = 10  # Measured in 'global steps'.

PAD = b"<pad>"
UNK = b"<unk>"
START_VOCAB = [PAD, UNK]
PAD_ID = 0
UNK_ID = 1

SERIALISATION_KEY_BATCH = 'batch'
SERIALISATION_KEY_EPOCH = 'epoch'
SERIALISATION_KEY_GLOBAL_STEP = 'gstep'
SERIALISATION_KEY_MODEL = 'model'
SERIALISATION_KEY_OPTIM = 'optim'
