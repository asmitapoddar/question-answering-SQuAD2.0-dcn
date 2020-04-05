from constants import *
from model import *
from tqdm import tqdm
import json
import os
import sys
import torch as th

th.manual_seed(RANDOM_SEED)

# Dimensionality of word vectors in glove.840B.300d (also in glove.6B.300d) = 300 (EMBEDDING_DIM)

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
			word = line[:-(len(' '.join(values[-EMBEDDING_DIM:]))+2)]
			tmp = list(map(float, values[-EMBEDDING_DIM:]))
			coefs = th.tensor(tmp)
			embeddings_index[word] = coefs
	return embeddings_index

def encode_word(word, embeddings):
	# Set embeddings for out of vocabulary words to zero
	if word in embeddings: 
		return embeddings[word]
	else:
		return th.zeros(EMBEDDING_DIM)

def encode_token_list(embeddings, token_list, pad_length):
	word_vectors = th.zeros(0, EMBEDDING_DIM)
	for token in token_list:
		vec = encode_word(token, embeddings).unsqueeze(1).transpose(0, 1)
		word_vectors = th.cat((word_vectors, vec), dim=0)

	length_diff = pad_length - word_vectors.size()[0]
	if length_diff > 0:
		word_vectors = th.cat((word_vectors, th.zeros((length_diff, EMBEDDING_DIM))), dim=0)

	return word_vectors


def build_forward_input(embeddings, dataset_tokenized, evaluation_batch_size):
	data = dataset_tokenized["data"]	
	# batch[0] document embeddings  
	# batch[1] question embeddings
	# batch[2] question identifiers
	# batch[3] document strings
	# batch[4] document (token, start pos, end pos) list
	# batch[5] question string
	batch = ([], [], [], [], [], [])

	for item in tqdm(data):
		for para in item["paragraphs"]:

			context = para["context"]

			# list of (token string, start index, end index)
			context_enriched = para["context_tokens"]

			just_context_tokens = list(map(lambda x : x[0], context_enriched)) 

			context_embeddings = encode_token_list(embeddings, just_context_tokens, MAX_CONTEXT_LEN)

			for qas in para["qas"]:

				question = qas["question"]
				
				question_enriched = qas["question_tokens"]
				just_question_tokens = list(map(lambda x : x[0], question_enriched))
				question_embeddings = encode_token_list(embeddings, just_question_tokens, MAX_QUESTION_LEN)

				# Unique identifier for (question, corresponding answers)
				qas_id = qas["id"]

				batch[0].append(context_embeddings)
				batch[1].append(question_embeddings)
				batch[2].append(qas_id)
				batch[3].append(context)
				batch[4].append(context_enriched)
				batch[5].append(question)

				if len(batch[2]) == evaluation_batch_size:
					yield batch
					batch = ([], [], [], [], [], [])
	if len(batch[2]) > 0:
		yield batch

def load_model_for_evaluation(state_file_path, device):
    if state_file_path is not None:
        if not os.path.isfile(state_file_path):
            print("Failed to read path %s, aborting." % state_file_path)
            sys.exit()
        if DISABLE_CUDA:
        	state = th.load(state_file_path, map_location=th.device('cpu'))
        else:
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

# load the dev set after tokenization
def load_dev_set(eval_set_path):
	with open(eval_set_path, "r") as f:
		dev_set = json.load(f)
	return dev_set

def debugSurroudingWords(s, e, context_tokens, num=1):
	snew = s - num
	enew = e + num
	snew = max(0, snew)
	enew = min(len(context_tokens) - 1, enew)
	print("s:%d -> %d\ne:%d -> %d\n" % (s, snew, e, enew))
	return snew,enew

def run_evaluation(model_path, eval_set_path, output_path, shouldDebugSurroudingWords = False):

	print("Producing answers for:\nModel: %s\nFile: %s\nOutput path:%s\nDebug surrounding words:%s\n" % (model_path, eval_set_path, output_path, shouldDebugSurroudingWords))

	# TODO: use non-trivial batching for evaluation?
	evaluation_batch_size = 1

	# https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/2
	# Is GPU available:
	print ("cuda device count = %d" % th.cuda.device_count())
	print ("cuda is available = %d" % th.cuda.is_available())
	device = th.device("cuda:0" if th.cuda.is_available() and (not DISABLE_CUDA) else "cpu")

	model = load_model_for_evaluation(model_path, device)
	model.eval()

	dev_set_tokenized = load_dev_set(eval_set_path)

	# Load glove word vectors into a dictionary
	glove = load_embeddings_index()
	batch_iterator = build_forward_input(glove, dev_set_tokenized, evaluation_batch_size)

	# The file that will be provided to evaluate-v2.0.py
	answer_mapping = {}

	for batch in tqdm(batch_iterator):
		print("\n")

		context_vectors, question_vectors, context_ids, context_paras, context_enriched, questions = batch
		context_vectors = context_vectors[0].unsqueeze(dim=0).to(device)
		question_vectors = question_vectors[0].unsqueeze(dim=0).to(device)
		
		assert(context_vectors.size()[1+1] == EMBEDDING_DIM)
		assert(question_vectors.size()[1+1] == EMBEDDING_DIM) 
		
		# Fake ground truth data (one batch of starts and ends):
		true_s = th.randint(0, context_vectors.size()[1], (evaluation_batch_size,), device=device)
		true_e = th.randint(0, question_vectors.size()[1], (evaluation_batch_size,), device=device)
		for i in range(evaluation_batch_size):
			true_s[i], true_e[i] = min(true_s[i], true_e[i]), max(true_s[i], true_e[i])

		# Run model
		_, s, e = model.forward(context_vectors, question_vectors, true_s, true_e)

		context_token_list = context_paras[0]

		if shouldDebugSurroudingWords:
			s, e = debugSurroudingWords(s, e, context_enriched[0], num=1)

		# We need this because the model could output spans that aren't within
		# the document (we zero pad and add a sentinel)
		num_tokens_excl_padding_and_sentinel = len(context_enriched[0]) - 1
		s = min(s, num_tokens_excl_padding_and_sentinel)
		e = min(e, num_tokens_excl_padding_and_sentinel)

		ansStartTok = context_enriched[0][s]
		ansStartIdx = ansStartTok[1]

		ansEndTok = context_enriched[0][e]
		ansEndIdx = ansEndTok[2]		

		answerSubstring = context_paras[0][ansStartIdx:ansEndIdx]
		print("id=%s\nquestion=%s\n" % (context_ids[0], questions[0]))
		print("start=%d\n end=%d\n substring=%s\n" % (ansStartIdx, ansEndIdx, answerSubstring))
		
		answer_mapping[context_ids[0]] = answerSubstring

	with open(output_path, "w") as f:
		json.dump(answer_mapping, f)

# TODO: provide path to serialised model
saved_state_path = None if len(sys.argv) <= 1 else sys.argv[1]
evaluation_set_path = "preprocessing/data/dev-v2.0-tokenized.json" if len(sys.argv) <= 2 else sys.argv[2]
predictions_output_path = "predictions.json" if len(sys.argv) <= 3 else sys.argv[3]
run_evaluation(saved_state_path, evaluation_set_path, predictions_output_path)
