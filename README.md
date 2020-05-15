# Dynamic Coattention Networks ForQuestion Answering

This project aims at implementing a Dynamic Coattention Network proposed by [Xionget al.(2017)](https://arxiv.org/abs/1611.01604) for Question Answering, learning to find answers spans in a document, given a question from the [Stanford Question Answering Dataset (SQuAD2.0)](https://rajpurkar.github.io/SQuAD-explorer/), using the **PyTorch Framework**. 

Inspite of the thrust on computer vision for medical applications, it is not widely adapted in real-life. Our aim is to build trust on the deep learning system, and deploy such a system in the following ways:   
- Apart from achieving high accuracy in predicting the class of DR using Convolutional Neural Networks, we estimate the uncertainty of neural networks in making its prediction. The deep learning system should give high confidence predictions when the predictions are likely to be correct and low confidence when the system is unsure.   
- We also generate visual explanation of the deep learning system to convey the pixels in the image that influences its decision. For a visual explanation to enhance trust, it has to be specific and relevant. It should only highlight the parts of image that is most relevant to how human justify its decision.  
- Create an end-to-end application which enables an end-user (such as a clinician) to obtain all the results on a dashboard to interpret model predictions. Deep-learning systems could thus, aid physicians by offering second opinions and flagging concerning areas in images.  


### Resuming training from saved state
* `python3 training_pipeline.py "model/2020-03-28_22-39-28/epoch0_batch11.par"`

### Generate scores for a model at different stages throughout its training
* `python3 gen_scores.py <model_path> <dataset_file_path.json> [optional eval freq.] [optional eval start step]`
* Concrete example (training set): `python3 gen_scores.py ./model/MI1_dropout_encodings_only/ preprocessing/data/subset-1/train-subset-1.json 2000 50000` -- this will eval model at step 50000, 52000, 54000, ... up to the most recent one.
* Concrete example (dev set): `python3 gen_scores.py ./model/MI1_dropout_encodings_only/ preprocessing/data/dev-v2.0.json`
* *The dataset file path needs to be `something.json` and have a corresponding `something-tokenized.json` for this script to work!*
* The script will generate a file `scores_<datasetname>.log` in the model folder, as well as two plots (EM and F1).
* To copy the plots to your computer run: `scp -T guest@138.19.43.95:"'Documents/no_eating_no_drinking/model/MI1_dropout_encodings_only/plot_loss_vs_em_score(train-subset-1).png'" . && scp -T guest@138.19.43.95:"'Documents/no_eating_no_drinking/model/MI1_dropout_encodings_only/plot_loss_vs_f1_score(train-subset-1).png'" . && ` (or same but with `dev-v2` replacing `train-subset-1`).

### Produce answer file for evaluation
* Generate predictions on **SQuAD dev set**: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par`
* Generate predictions on a different dataset: `python3 produce_answers.py model/2020-04-01_01-07-06/epoch0_batch791.par preprocessing/data/subset-1/train-subset-1-tokenized.json [optional_prediction_file_path]`
* Run evaluation: `python3 evaluate-v2.0.py  preprocessing/data/subset-1/train-subset-1.json predictions.json`

### Plot F1 score and loss together
* First generate the scores log file using `gen_scores.py` (see separate instructions for that).
* Then: `python3 plot_f1_vs_loss.py model/mymodel/scores_train-subset-1.log`. 

### Existing impl (atulkum)
* Model: https://github.com/atulkum/co-attention/blob/master/code/model.py

### Colab training setup
* GDrive: https://drive.google.com/drive/folders/1n5V3Je-qcuncPhkDYikIbAxJ_68d1p4E
* Colab: https://colab.research.google.com/drive/1ycVllF_XIsXDvC4qOAMr4HRTeRaEBFNb

Xiong, Caiming, Zhong, Victor, & Socher, Richard. 2017.  Dynamic Coattention Networks for Question Answering, International Conference on Learning Representations (ICLR) 
