from constants import *
import os
from pathlib import Path
import subprocess
import sys
import torch as th
    
FREQ = 200  # Evals every 200 global steps.
TEMP_JSON_FILENAME = "kuba_temp.json"

def gen_predictions(model_path, dataset_path):
    tokenized_dataset_path = dataset_path.split(".")[0]+"-tokenized.json"
    cmd = ["python3", "produce_answers.py", model_path, tokenized_dataset_path, TEMP_JSON_FILENAME]
    print("Calling subprocess '%s'..." % " ".join(list(map(str,cmd))))
    subprocess.run(cmd, check=True)

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
    if len(sys.argv) != 3:
        print()
    else:
        model_dir = sys.argv[1]
        if model_dir[-1] != '/':
            model_dir += '/'
        dataset_path = sys.argv[2]

        next_global_step_to_eval = 0

        with open(model_dir+"scores.log", "w") as scores_file:
            model_paths = sorted(Path(model_dir).iterdir(), key=os.path.getmtime)
            model_paths = [p for p in model_paths if ".par" in str(p)]
            for i, model_path in enumerate(model_paths):
                assert(".par" in str(model_path))
                global_step = th.load(model_path)[SERIALISATION_KEY_GLOBAL_STEP]
                if global_step >= next_global_step_to_eval:
                    next_global_step_to_eval += FREQ
                    print("Evaluating model: '%s' ..." % model_path, flush=True)
                    gen_predictions(model_path, dataset_path)
                    em_score, f1_score, total = run_eval(dataset_path)
                    scores_file.write(global_step + "," + em_score + "," + f1_score + "," + total + "\n")
