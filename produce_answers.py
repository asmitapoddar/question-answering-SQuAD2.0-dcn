from model import *
from tqdm import tqdm
import json
import sys

import torch as th

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
	batch = ([], [], [], [])
	for item in tqdm(data): 
		for para in item["paragraphs"]:
			context = para["context"]

			context_embeddings = encode_token_list(embeddings, context)
			para["context_embedding"] = context_embeddings

			for qas in para["qas"]:
				question = qas["question"]
				question_embeddings = encode_token_list(embeddings, question)
				qas["question_embedding"] = question_embeddings

				# Unique identifier for (question, corresponding answers)
				qas_id = qas["id"]

				batch[0].append(context_embeddings)
				batch[1].append(question_embeddings)
				batch[2].append(qas_id)
				batch[3].append(context)

				if len(batch[2]) == evaluation_batch_size:
					yield batch
					batch = ([], [], [], [])

	if len(batch[2]) > 0:
		yield batch


def run_evaluation():
	# Load glove word vectors into a dictionary 
	glove = load_embeddings_index(small=False)

	# TODO: use non-trivial batching for evaluation?
	evaluation_batch_size = 1
	batch_iterator = build_forward_input(glove, dev_set_tokenized, evaluation_batch_size)

	# https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/2
	# Is GPU available:
	print ("cuda device count = %d" % th.cuda.device_count())
	print ("cuda is available = %d" % th.cuda.is_available())
	device = th.device("cuda:0" if th.cuda.is_available() and (not TEST_DCN_MODEL_WITH_CPU) else "cpu")

	# The file that will be provided to evaluate-v2.0.py
	answer_mapping = {}

	for batch in tqdm(batch_iterator):

		context_vectors, question_vectors, context_ids, context_paras = batch
		context_vectors = context_vectors[0].unsqueeze(dim=0)
		question_vectors = question_vectors[0].unsqueeze(dim=0)
		
		assert(context_vectors.size()[1+1] == DIMENSIONALITY)
		assert(question_vectors.size()[1+1] == DIMENSIONALITY) 
		
		# Fake ground truth data (one batch of starts and ends):
		true_s = th.randint(0, context_vectors.size()[1], (evaluation_batch_size,), device=device)
		true_e = th.randint(0, question_vectors.size()[1], (evaluation_batch_size,), device=device)
		for i in range(evaluation_batch_size):
		  true_s[i], true_e[i] = min(true_s[i], true_e[i]), max(true_s[i], true_e[i])

		# Run model.
		model = DCNModel(context_vectors, question_vectors, evaluation_batch_size, device).to(device)
		loss, s, e = model.forward(context_vectors, question_vectors, true_s, true_e)
		answer_substring = " ".join(context_paras[0][s:e])
		answer_mapping[context_ids[0]] = answer_substring

	with open("predictions.json", "w") as f:
		json.dump(answer_mapping, f)

run_evaluation()
