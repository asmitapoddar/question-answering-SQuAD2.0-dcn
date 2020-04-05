**Make sure your pushes to master doesn't break any tests.**

# Dynamic Coattention Networks For Question Answering

This repository contain a reimplementation of this paper https://arxiv.org/abs/1611.01604.

The repo name speaks for itself.

### Resuming training from saved state
* `python3 training_pipeline.py "/home/no_eating_no_drinking/model/2020-03-28_22-39-28/epoch0_batch11.par"`

### Produce answer file for evaluation
* Generate predictions on **SQuAD dev set**: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par`
* Generate predictions on a different dataset: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par preprocessing/data/subset-1/train-subset-1-tokenized.json [optional_prediction_file_path]`
* Run evaluation: `python3 evaluate-v2.0.py  preprocessing/data/subset-1/train-subset-1.json predictions.json`

### Existing impl
* Model: https://github.com/atulkum/co-attention/blob/master/code/model.py
* Batcher: https://github.com/atulkum/co-attention/blob/master/code/data_util/data_batcher.py

### Links to our content
* GDrive: https://drive.google.com/open?id=17K0ZFb_OCdvHgSlkNFErzyx--eYZnoiG
* AML GDoc: https://docs.google.com/document/d/1fit7dYVHn0I0PsAA_HCj3AqnxJ7Wzz-78sb--wxKdA4/edit?usp=sharing
* Group Report: https://www.overleaf.com/4391458899tptsptshghhx

### TODOs
- [ ] ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) *Add your **past** contributions or **nearest-future work** here. (everyone)*
- [x] Move tests to seperate file (Richie)
- [x] Implement model (everyone)
- [x] Enable cuda usage (Kuba)
- [x] Get forward pass to run (Kuba)
- [x] Get backward pass to run (Kuba -- this was quick)
- [x] Debug why predicted end indices are all 0 (Richie)
- [x] Complete batching (Asmita)
- [x] Create word2id, id2word, embedding matrix (Asmita)
- [x] Training pipeline (Asmita + Kuba's minor cleanup)
- [x] Model serialisation (Kuba + Richie)
- [x] Debug `retain_graph` error (Dip)
- [x] Debug training issues (Dip with help from Kuba and Richie)
- [ ] Ablation tests:
  - [ ] single iteration for s/e indices instead of 4. On a related note, log the iteration convergence behaviour.
  - [ ] smaller HIDDEN_DIM
  - [ ] try removing some modules or replacing them with something simpler, e.g. coattention with some fully connected layers.
  - [ ] *Think of more ablation tests.*
- [ ] Generate predictions for evaluation (TODO batching if needed, better conversion from tokens to answer strings, ~~load serialised model~~) (Dip)
