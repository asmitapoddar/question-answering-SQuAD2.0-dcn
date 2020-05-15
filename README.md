# Dynamic Coattention Networks For Question Answering

This project aims at implementing a Dynamic Coattention Network proposed by [Xionget al.(2017)](https://arxiv.org/abs/1611.01604) for Question Answering, learning to find answers spans in a document, given a question from the [Stanford Question Answering Dataset (SQuAD2.0)](https://rajpurkar.github.io/SQuAD-explorer/), using the **PyTorch Framework**. Performance is evaluated with the standard tokenwise F1 score and EM (exact match) percentage over the predicted answers

## Code
The following scripts, stored in this repository, have been developed for implementing Factorization Machines for music recommendation using the dataset.
1. 
The DCN is comprised of two components - the Coattention Encoder and the Dynamic Pointer Decoder  

## Usage

### TrainingR
* esuming training from saved state: `python3 training_pipeline.py "model/2020-03-28_22-39-28/epoch0_batch11.par"`

### Generate scores for a model at different stages throughout its training
* Generating scores for a model: `python3 gen_scores.py <model_path> <dataset_file_path.json> [optional eval freq.] [optional eval start step]`. 
Example (training set): `python3 gen_scores.py ./model/MI1_dropout_encodings_only/ preprocessing/data/subset-1/train-subset-1.json 2000 50000` -- this will eval model at step 50000, 52000, 54000, ... up to the most recent one.  
Example (dev set): `python3 gen_scores.py ./model/MI1_dropout_encodings_only/ preprocessing/data/dev-v2.0.json`
* *The dataset file path needs to be `something.json` and have a corresponding `something-tokenized.json` for this script to work!*
* The script will generate a file `scores_<datasetname>.log` in the model folder, as well as two plots (EM and F1).

### Produce answer file for evaluation
* Generate predictions on **SQuAD dev set**: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par`
* Generate predictions on a different dataset: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par preprocessing/data/subset-1/train-subset-1-tokenized.json [optional_prediction_file_path]`
* Run evaluation: `python3 evaluate-v2.0.py  preprocessing/data/subset-1/train-subset-1.json predictions.json`

### Plot F1 score and loss together
* First generate the scores log file using `gen_scores.py` (see separate instructions for that).
* Then: `python3 plot_f1_vs_loss.py model/mymodel/scores_train-subset-1.log`. 


Xiong, Caiming, Zhong, Victor, & Socher, Richard. 2017.  Dynamic Coattention Networks for Question Answering, International Conference on Learning Representations (ICLR) 
