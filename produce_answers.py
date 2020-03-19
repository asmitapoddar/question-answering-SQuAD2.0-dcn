from model import *
from tqdm import tqdm

# Load glove word vectors into a dictionary 

# Dimensionality of word vectors in glove.840B.300d (also in glove.6B.300d)
DIMENSIONALITY = 300

def load_embeddings_index(small = False):
    embeddings_index = {}

    filePath = "preprocessing/glove/"
    if small:
        filePath += "glove.6B.300d.txt"
    else:
        filePath += "glove.840B.300d.txt"

    print("Loading embeddings from " + filePath)

    with open(filePath, 'r', encoding="utf8") as f:
        for line in tqdm(f):
            values = line.split()
            word = line[:-(len(' '.join(values[-DIMENSIONALITY:]))+2)]
            coefs = np.asarray(values[-DIMENSIONALITY:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


glove = load_embeddings_index()

# Set out of vocab word vectors to zero
# Extract the question id 

# https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/2
# Is GPU available:
print ("cuda device count = %d" % th.cuda.device_count())
print ("cuda is available = %d" % th.cuda.is_available())
device = th.device("cuda:0" if th.cuda.is_available() and (not TEST_DCN_MODEL_WITH_CPU) else "cpu")

doc = th.randn(BATCH_SIZE, 30, HIDDEN_DIM, device=device) # Fake word vec dimension set to HIDDEN_DIM.
que = th.randn(BATCH_SIZE, 5, HIDDEN_DIM, device=device)  # Fake word vec dimension set to HIDDEN_DIM.

# Fake ground truth data (one batch of starts and ends):
true_s = th.randint(0, doc.size()[1], (BATCH_SIZE,), device=device)
true_e = th.randint(0, doc.size()[1], (BATCH_SIZE,), device=device)
for i in range(BATCH_SIZE):
  true_s[i], true_e[i] = min(true_s[i], true_e[i]), max(true_s[i], true_e[i])

# Run model.
model = DCNModel(doc, que, BATCH_SIZE, device).to(device)
loss, s, e = model.forward(doc, que, true_s, true_e)
print("Predicted start: %s \nPredicted end: %s \nloss: %s" % (str(s), str(e), str(loss)))
model.zero_grad()
loss.backward()

print("%d/%d parameters have non-None gradients." % (len([param for param in model.parameters() if param.grad is not None]), len(list(model.parameters()))))