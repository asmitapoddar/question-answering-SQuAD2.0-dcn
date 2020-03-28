# Defaults/constants.

BATCH_SIZE = 64
DISABLE_CUDA = False # If this is set to True, make sure it should be!
DROPOUT = 0.3
EMBEDDING_DIM = 300
HIDDEN_DIM = 201  # Denoted by 'l' in the paper.
MAX_ITER = 4
MAXOUT_POOL_SIZE = 15
MAX_CONTEXT_LEN = 600
MAX_GRAD_NORM = 0.5
MAX_QUESTION_LEN = 30
NUM_EPOCHS = 1000 
REG_LAMBDA = 0.1

PAD = b"<pad>"
UNK = b"<unk>"
START_VOCAB = [PAD, UNK]
PAD_ID = 0
UNK_ID = 1
