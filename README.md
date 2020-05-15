# Dynamic Coattention Networks For Question Answering

This project aims at implementing a Dynamic Coattention Network proposed by [Xionget al.(2017)](https://arxiv.org/abs/1611.01604) for Question Answering, learning to find answers spans in a document, given a question from the [Stanford Question Answering Dataset (SQuAD2.0)](https://rajpurkar.github.io/SQuAD-explorer/), using the **PyTorch Framework**. Performance is evaluated with the standard tokenwise F1 score and EM (exact match) percentage over the predicted answers.

Several deep learning models have been proposed for question answering. However, due to their single-pass nature, they have no way to recover from local maxima corresponding to incorrect answers. To address this problem, we introduce
the Dynamic Coattention Network (DCN) for question answering. The DCN first fuses co-dependent representations of the question and the document in order to focus on relevant parts of both. Then a dynamic pointing decoder iterates over potential answer spans. This iterative procedure enables the model to recover from initial local maxima corresponding to incorrect answers.


## Code
The following scripts, stored in this repository, have been developed for implementing Dynamic Coattention Networks for Question Answering using the SQuAD dataset.
1. [preprocessing](https://github.com/asmitapoddar/question-answering-dcn/tree/master/preprocessing): Preprocessing done on the dataset including tokenizing the data, loading the GloVE embeddings, getting the embeddings for the data and batching the data for training.    
2. [model.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/model.py): The Dynamic Coattention Network model, which comprises two components - the Coattention Encoder and the Dynamic Pointer Decoder.  
3. [constants.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/constants.py): Hyper-paramters of the model.  
4. [training_pipeline.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/training_pipeline.py): Script to train the model.  
5. [produce_answers.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/produce_answers.py): Script to produce the answers of the given dataset using the model (path) provided.    
6. [gen_scores.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/gen_scores.py): Script to generate scores *todo*. 
7. [evaluate-v2.0.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/evaluate-v2.0.py): Evaluation script to find F1 and EM score of the model.   
8. [test_model.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/test_model.py): Testing the model using dummy values (for debugging the model).    
9. [gen_plot.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/gen_plot.py):   
10. [gen_f1_by_length.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/gen_f1_by_length.py):   
11. [index_convergence.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/index_convergence.py):   
12. [make_plot_f1.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/make_plot_f1.py):   
13. [plot_f1_vs_loss.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/plot_f1_vs_loss.py):   
14. [training_pipeline_freeze_nondpd_weights.py](https://github.com/asmitapoddar/question-answering-dcn/blob/master/training_pipeline_freeze_nondpd_weights.py):   

## Usage

### Training
* Training model: `python3 training_pipeline.py`
* Resuming training from saved state: `python3 training_pipeline.py "model/2020-03-28_22-39-28/epoch0_batch11.par"`

### Generate scores for a model at different stages throughout its training
* Generating scores for a model: `python3 gen_scores.py <model_path> <dataset_file_path.json> [optional eval freq.] [optional eval start step]`.   
Example (training set): `python3 gen_scores.py ./model/MI1_dropout_encodings_only/ preprocessing/data/subset-1/train-subset-1.json 2000 50000` -- this will eval model at step 50000, 52000, 54000, ... up to the most recent one.   
Example (dev set): `python3 gen_scores.py ./model/MI1_dropout_encodings_only/ preprocessing/data/dev-v2.0.json`. 

Note: The dataset file path needs to be `something.json` and have a corresponding `something-tokenized.json` for this script to work!
The script will generate a file `scores_<datasetname>.log` in the model folder, as well as two plots (EM and F1).

### Produce answer file for evaluation
* Generate predictions on *SQuAD dev set*: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par`
* Generate predictions on a different dataset: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par preprocessing/data/subset-1/train-subset-1-tokenized.json [optional_prediction_file_path]`
* Run evaluation: `python3 evaluate-v2.0.py  preprocessing/data/subset-1/train-subset-1.json predictions.json`

### Plot F1 score and loss together
* First generate the scores log file using `gen_scores.py` (see separate instructions for that).
* Then: `python3 plot_f1_vs_loss.py model/mymodel/scores_train-subset-1.log`. 


Xiong, Caiming, Zhong, Victor, & Socher, Richard. 2017.  Dynamic Coattention Networks for Question Answering, International Conference on Learning Representations (ICLR) 
