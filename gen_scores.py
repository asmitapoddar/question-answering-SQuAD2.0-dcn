from constants import *
import os
from pathlib import Path
from produce_answers import load_embeddings_index, run_evaluation
import subprocess
import sys
import torch as th
    
DEFAULT_FREQ = 200  # Evals every 200 global steps.
TEMP_JSON_FILENAME = "kuba_temp.json"

def gen_predictions(model_path, dataset_path, glove):
    tokenized_dataset_path = dataset_path.split(".")[0]+"-tokenized.json"
    print("Calling produce_answers.run_evaluation()...")
    run_evaluation(str(model_path), tokenized_dataset_path, TEMP_JSON_FILENAME, glove)

def run_eval(dataset_path):
    cmd = ["python3", "evaluate-v2.0.py", dataset_path, TEMP_JSON_FILENAME]
    print("Calling subprocess '%s'..." % " ".join(list(map(str,cmd))))
    p = subprocess.run(["python3", "evaluate-v2.0.py", dataset_path, TEMP_JSON_FILENAME], check=True, stdout=subprocess.PIPE, universal_newlines=True)
    outp = p.stdout
    em_score = outp.split('"exact": ')[1].split(",")[0]
    f1_score = outp.split('"f1": ')[1].split(",")[0]
    total = outp.split('"total": ')[1].split(",")[0]
    return em_score, f1_score, total

if __name__ == "__main__":
    if len(sys.argv) not in [3,4]:
        print("Usage example: \npython3 gen_scores.py ./model/2020-04-07_00-10-37\[LR1-00e-03_Q86821_B64_H200_RS1\]/ preprocessing/data/subset-4/train-subset-4.json")
        print("Can add optional FREQ parameter at the end (integer regulating frequency of evaluation measured in global steps).")
    else:
        glove = load_embeddings_index() # Do this once.

        model_dir = sys.argv[1]
        if model_dir[-1] != '/':
            model_dir += '/'
        dataset_path = sys.argv[2]
        freq = DEFAULT_FREQ if len(sys.argv)==3 else sys.argv[3]

        next_global_step_to_eval = 0
        if os.path.exists(model_dir+"scores.log"):
            print("scores.log already exists. Do you want to overwrite it?[y/n]")
            user_ans = str(input())
            if "y" in user_ans:
                with open(model_dir+"scores.log", "w") as scores_file:
                    model_paths = sorted(Path(model_dir).iterdir(), key=os.path.getmtime)
                    model_paths = [p for p in model_paths if ".par" in str(p)]
                    for i, model_path in enumerate(model_paths):
                        assert(".par" in str(model_path))
                        global_step = th.load(model_path)[SERIALISATION_KEY_GLOBAL_STEP]
                        if global_step >= next_global_step_to_eval:
                            next_global_step_to_eval += freq
                            print("Evaluating model: '%s' ..." % model_path, flush=True)
                            gen_predictions(model_path, dataset_path, glove)
                            em_score, f1_score, total = run_eval(dataset_path)
                            scores_file.write(str(global_step) + "," + str(em_score) + "," + str(f1_score) + "," + str(total) + "\n")
