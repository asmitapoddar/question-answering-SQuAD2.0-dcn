from model import *
from tqdm import tqdm
import json
import sys
import torch as th
import os
from constants import *

th.manual_seed(1)

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
			tmp = list(map(float, values[-DIMENSIONALITY:]))
			coefs = th.tensor(tmp)
			embeddings_index[word] = coefs
	return embeddings_index

# load the dev set after tokenization
def load_dev_set(): 
	with open("preprocessing/data/dev-v2.0-tokenized.json", "r") as f:
		dev_set = json.load(f)
	return dev_set

dev_set_tokenized = load_dev_set()

def encode_word(word, embeddings):
	# Set embeddings for out of vocabulary words to zero
	if word in embeddings: 
		return embeddings[word]
	else:
		return th.zeros(DIMENSIONALITY)

def encode_token_list(embeddings, token_list):
	word_vectors = th.zeros(0, DIMENSIONALITY)
	for token in token_list:
		vec = encode_word(token, embeddings).unsqueeze(1).transpose(0, 1)
		word_vectors = th.cat((word_vectors, vec), dim=0)
	return word_vectors


def build_forward_input(embeddings, dataset_tokenized, evaluation_batch_size):
	data = dataset_tokenized["data"]	
	# batch[0] document embeddings  
	# batch[1] question embeddings
	# batch[2] question identifiers
	# batch[3] document strings
	# batch[4] document (token, start pos, end pos) list
	batch = ([], [], [], [], [])

	for item in tqdm(data):
		for para in item["paragraphs"]:

			context = para["context"]

			# list of (token string, start index, end index)
			context_enriched = para["context_tokens"]

			just_context_tokens = list(map(lambda x : x[0], context_enriched)) 

			context_embeddings = encode_token_list(embeddings, just_context_tokens)

			for qas in para["qas"]:

				question = qas["question"]
				
				question_enriched = qas["question_tokens"]
				just_question_tokens = list(map(lambda x : x[0], question_enriched))
				question_embeddings = encode_token_list(embeddings, just_question_tokens)

				# Unique identifier for (question, corresponding answers)
				qas_id = qas["id"]

				batch[0].append(context_embeddings)
				batch[1].append(question_embeddings)
				batch[2].append(qas_id)
				batch[3].append(context)
				batch[4].append(context_enriched)

				if len(batch[2]) == evaluation_batch_size:
					yield batch
					batch = ([], [], [], [], [])
	if len(batch[2]) > 0:
		yield batch

def load_model_for_evaluation(state_file_path, device):
    if state_file_path is not None:
        if not os.path.isfile(state_file_path):
            print("Failed to read path %s, aborting." % state_file_path)
            sys.exit()
        state = th.load(state_file_path)
        if len(state) != 5:
            print("Invalid state read from path %s, aborting. State keys: %s" % (state_file_path, state.keys()))
            sys.exit()
        model = DCNModel(1, device).to(device)
        model.load_state_dict(state[SERIALISATION_KEY_MODEL])
        return model
    else:
    	print("No model state path provided, aborting.")
    	sys.exit()



def run_evaluation(model_path, output_path = "predictions.json"):

	# Load glove word vectors into a dictionary 
	# TODO: Change to 840B word embeddings
	glove = load_embeddings_index()
	
	# TODO: use non-trivial batching for evaluation?
	evaluation_batch_size = 1
	batch_iterator = build_forward_input(glove, dev_set_tokenized, evaluation_batch_size)

	# https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/2
	# Is GPU available:
	print ("cuda device count = %d" % th.cuda.device_count())
	print ("cuda is available = %d" % th.cuda.is_available())
	device = th.device("cuda:0" if th.cuda.is_available() and (not DISABLE_CUDA) else "cpu")

	# The file that will be provided to evaluate-v2.0.py
	answer_mapping = {}

	model = load_model_for_evaluation(model_path, device)
	model.eval()

	for batch in tqdm(batch_iterator):

		context_vectors, question_vectors, context_ids, context_paras, context_enriched = batch
		context_vectors = context_vectors[0].unsqueeze(dim=0).to(device)
		question_vectors = question_vectors[0].unsqueeze(dim=0).to(device)
		
		assert(context_vectors.size()[1+1] == DIMENSIONALITY)
		assert(question_vectors.size()[1+1] == DIMENSIONALITY) 
		
		# Fake ground truth data (one batch of starts and ends):
		true_s = th.randint(0, context_vectors.size()[1], (evaluation_batch_size,), device=device)
		true_e = th.randint(0, question_vectors.size()[1], (evaluation_batch_size,), device=device)
		for i in range(evaluation_batch_size):
		  true_s[i], true_e[i] = min(true_s[i], true_e[i]), max(true_s[i], true_e[i])

		# Run model
		_, s, e = model.forward(context_vectors, question_vectors, true_s, true_e)

		ansStartTok = context_enriched[0][s]
		ansStartIdx = ansStartTok[1]

		ansEndTok = context_enriched[0][e]
		ansEndIdx = ansEndTok[2]
		answerSubstring = context_paras[0][ansEndIdx:ansStartIdx]
		print("start=%d, end=%d, substring=%s" % (ansStartIdx, ansEndIdx, answerSubstring))
		
		answer_mapping[context_ids[0]] = answerSubstring

	with open(output_path, "w") as f:
		json.dump(answer_mapping, f)

# TODO: provide path to serialised model
saved_state_path = None if len(sys.argv) <= 1 else sys.argv[1]
run_evaluation(saved_state_path)