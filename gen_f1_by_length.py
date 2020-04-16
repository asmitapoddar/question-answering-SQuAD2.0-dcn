"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import numpy as np
from make_plot_f1 import *
from produce_answers import load_embeddings_index, run_evaluation

OPTS = None
TEMP_JSON_FILENAME_F1_PLOT = "gen_f1_by_length_temp.json"

# Constants passed to plotting functions to indicate whether to
# plot stdev or confidence interval error bars
ERROR_BAR_TYPE_PERCENTILE = "ERROR_BAR_TYPE_PERCENTILE"
ERROR_BAR_TYPE_STDEV = "ERROR_BAR_TYPE_STDEV"
ERROR_BAR_PERCENTILE_VALUE = 95

DEFAULT_ERROR_BAR_TYPE = ERROR_BAR_TYPE_PERCENTILE



def parse_args():
  parser = argparse.ArgumentParser('Plot F1 scores and standard deviations by document/question/answer length.')
  parser.add_argument('model_params', metavar='modelparams.par', help='Model parameters.')
  parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
  parser.add_argument('out_image_path', metavar='outimage.svg', help='Image of F1 score plot across different lengths.')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # Skip unanswerable questions
          continue

          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def get_raw_scores_with_length_info(dataset, preds):
  exact_scores = {}
  f1_scores = {}

  ans_f1 = {}
  que_f1 = {}
  doc_f1 = {}
  produced_ans_f1 = {}

  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']

        # Average length of answer for this question
        average_gold_answer_length = int(np.rint(np.mean([len(get_tokens(ans)) for ans in gold_answers])))

        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        em = max(compute_exact(a, a_pred) for a in gold_answers)
        f1 = max(compute_f1(a, a_pred) for a in gold_answers)
        exact_scores[qid] = em
        f1_scores[qid] = f1

        qu_len = len(get_tokens(qa['question']))
        doc_len = len(get_tokens(p['context']))

        # Length of predicted answer
        produced_ans_len = len(get_tokens(a_pred))

        if str(average_gold_answer_length) not in ans_f1:
          ans_f1[str(average_gold_answer_length)] = []
        if str(qu_len) not in que_f1:
          que_f1[str(qu_len)] = []
        if str(doc_len) not in doc_f1:
          doc_f1[str(doc_len)] = []
        if str(produced_ans_len) not in produced_ans_f1:
          produced_ans_f1[str(produced_ans_len)] = []

        ans_f1[str(average_gold_answer_length)].append(f1)
        que_f1[str(qu_len)].append(f1)
        doc_f1[str(doc_len)].append(f1)
        produced_ans_f1[str(produced_ans_len)].append(f1)

  return ans_f1, que_f1, doc_f1, produced_ans_f1, f1_scores

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores

def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def plot_pr_curve(precisions, recalls, out_image, title):
  plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
  plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.title(title)
  plt.savefig(out_image)
  plt.clf()

def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  true_pos = 0.0
  cur_p = 1.0
  cur_r = 0.0
  precisions = [1.0]
  recalls = [0.0]
  avg_prec = 0.0
  for i, qid in enumerate(qid_list):
    if qid_to_has_ans[qid]:
      true_pos += scores[qid]
    cur_p = true_pos / float(i+1)
    cur_r = true_pos / float(num_true_pos)
    if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
      # i.e., if we can put a threshold after this point
      avg_prec += cur_p * (cur_r - recalls[-1])
      precisions.append(cur_p)
      recalls.append(cur_r)
  if out_image:
    plot_pr_curve(precisions, recalls, out_image, title)
  return {'ap': 100.0 * avg_prec}

def histogram_na_prob(na_probs, qid_list, image_dir, name):
  if not qid_list:
    return
  x = [na_probs[k] for k in qid_list]
  weights = np.ones_like(x) / float(len(x))
  plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
  plt.xlabel('Model probability of no-answer')
  plt.ylabel('Proportion of dataset')
  plt.title('Histogram of no-answer probability: %s' % name)
  plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
  plt.clf()

def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh

def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh

def compute_average_f1s(data, error_bar_type):
  points = []
  for len_str in data:
    f1s = data[len_str]
    points.append((int(len_str), np.mean(f1s), np.std(f1s)))
  return points

def plot_f1(ans_data, que_data, doc_data, outpath, error_bar_type):
  ans_len_avgf1_std = compute_average_f1s(ans_data, error_bar_type)
  que_len_avgf1_std = compute_average_f1s(que_data, error_bar_type)
  doc_len_avgf1_std = compute_average_f1s(doc_data, error_bar_type)
  make_plot_f1(ans_len_avgf1_std, que_len_avgf1_std, doc_len_avgf1_std, outpath)

def plot_f1_against_pred_len(pred_len_f1, outpath, error_bar_type):
  pred_len_avgf1_std = compute_average_f1s(pred_len_f1, error_bar_type)
  make_plot_f1_against_prediction_length(pred_len_avgf1_std, outpath)

def gen_predictions(model_path, dataset_path, glove):
    tokenized_dataset_path = ".".join(dataset_path.split(".")[:-1])+"-tokenized.json"
    print("Calling produce_answers.run_evaluation()...")
    run_evaluation(str(model_path), tokenized_dataset_path, TEMP_JSON_FILENAME_F1_PLOT, glove)

def main():
  glove = load_embeddings_index()
  gen_predictions(OPTS.model_params, OPTS.data_file, glove)

  with open(OPTS.data_file) as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']

  with open(TEMP_JSON_FILENAME_F1_PLOT) as f:
    preds = json.load(f)

  qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
  ans_f1, que_f1, doc_f1, pred_len_f1, f1_scores = get_raw_scores_with_length_info(dataset, preds)

  all_f1_scores = list(f1_scores.values())
  has_ans_f1_scores = [f1_scores[k] for k in has_ans_qids if k in f1_scores]
  no_ans_f1_scores = [f1_scores[k] for k in no_ans_qids if k in f1_scores]

  # Plot of F1 against length of doc/que/ans.
  plot_f1(ans_f1, que_f1, doc_f1, OPTS.out_image_path, DEFAULT_ERROR_BAR_TYPE)

  # Plot histogram of f1s.
  # Provide only the "HasAns" f1s.
  f1_outpath_name, f1_outpath_ext = os.path.splitext(OPTS.out_image_path)
  f1_outpath = f1_outpath_name + "_f1_histogram" + f1_outpath_ext
  plot_f1_histogram(has_ans_f1_scores, f1_outpath, DEFAULT_ERROR_BAR_TYPE)


  # Write summary file with percentage of F1 scores that are zero, one, or in between.
  f1_summary_outpath = f1_outpath_name + "_f1_dist_summary" + ".txt"
  f1_distribution_summary(has_ans_f1_scores, f1_summary_outpath)

  # Create plot of average F1 against produced-answer length
  f1_predicted_answer_length_outpath = f1_outpath_name + "_f1_against_prediction_len" + f1_outpath_ext
  plot_f1_against_pred_len(pred_len_f1, f1_predicted_answer_length_outpath, DEFAULT_ERROR_BAR_TYPE)

  # Delete temporary predictions file
  os.remove(TEMP_JSON_FILENAME_F1_PLOT)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
