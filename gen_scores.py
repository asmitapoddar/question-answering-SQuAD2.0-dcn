from constants import *
import os
from pathlib import Path
from produce_answers import load_embeddings_index, run_evaluation
import subprocess
import sys
import torch as th
    
DEFAULT_FREQ = 200  # Evals every 200 global steps.
TEMP_JSON_FILENAME = "gen_scores_temp.json" # Doesn't matter, just didn't want it to clash with other people's temp files.

def custom_print(x, flush=False):
    print("[gen_scores] ",end='',flush=flush)
    print(x)

def gen_predictions(model_path, dataset_path, glove):
    tokenized_dataset_path = ".".join(dataset_path.split(".")[:-1])+"-tokenized.json"
    custom_print("Calling produce_answers.run_evaluation()...")
    run_evaluation(str(model_path), tokenized_dataset_path, TEMP_JSON_FILENAME, glove)

def run_eval(dataset_path):
    cmd = ["python3", "evaluate-v2.0.py", dataset_path, TEMP_JSON_FILENAME]
    custom_print("Calling subprocess '%s'..." % " ".join(list(map(str,cmd))))
    p = subprocess.run(["python3", "evaluate-v2.0.py", dataset_path, TEMP_JSON_FILENAME], check=True, stdout=subprocess.PIPE, universal_newlines=True)
    outp = p.stdout
    em_score = outp.split('"exact": ')[1].split(",")[0]
    f1_score = outp.split('"f1": ')[1].split(",")[0]
    total = outp.split('"total": ')[1].split(",")[0]
    return em_score, f1_score, total

def main():
    if len(sys.argv) not in [3,4]:
        custom_print("Usage example: \npython3 gen_scores.py ./model/2020-04-07_00-10-37\[LR1-00e-03_Q86821_B64_H200_RS1\]/ preprocessing/data/subset-4/train-subset-4.json")
        custom_print("Can add optional FREQ parameter at the end (integer regulating frequency of evaluation measured in global steps).")
        return

    model_dir = sys.argv[1]
    if model_dir[-1] != '/':
        model_dir += '/'
    dataset_path = sys.argv[2]
    freq = DEFAULT_FREQ if len(sys.argv)==3 else int(sys.argv[3])

    scores_filename = "scores_"+dataset_path.split("/")[-1].split(".")[0]+".log"
    scores_path = model_dir+scores_filename

    next_global_step_to_eval = 0
    if os.path.exists(scores_path):
        custom_print("%s already exists. Do you want to overwrite it? [y/n] + Enter" % scores_filename)
        user_ans = str(input())
        if "y" not in user_ans:
            return
    with open(scores_path, "w") as scores_file:
        model_paths = sorted(Path(model_dir).iterdir(), key=os.path.getmtime)
        model_paths = [p for p in model_paths if ".par" in str(p)]

        glove = load_embeddings_index() # Do this once. Takes a few minutes!
    
        for i, model_path in enumerate(model_paths):
            assert(".par" in str(model_path))
            global_step = th.load(model_path)[SERIALISATION_KEY_GLOBAL_STEP]
            if global_step >= next_global_step_to_eval:
                next_global_step_to_eval += freq
                custom_print("Evaluating model: '%s' ..." % model_path,flush=True)
                gen_predictions(model_path, dataset_path, glove)
                em_score, f1_score, total = run_eval(dataset_path)
                scores_file.write(str(global_step) + "," + str(em_score) + "," + str(f1_score) + "," + str(total) + "\n")
    print("gen_scores finished. Log written to:", scores_path)


if __name__ == "__main__":
    main()
