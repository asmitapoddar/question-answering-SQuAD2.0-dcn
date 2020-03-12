import json
from tqdm import tqdm 

## Creates files for each group of data


def load_tokenize_train_set(): 
    with open("data/train-v2.0-tokenized.json", "r") as f:
        train_set = json.load(f)
    return train_set

def load_tokenize_dev_set(): 
    with open("data/dev-v2.0-tokenized.json", "r") as f:
        dev_set = json.load(f)
    return dev_set

def preprocess(dataset, type):
    # Takes the parsed JSON dataset, and replaces the questions and context documents 
    # by a sequence of word embeddings after tokenisation.

    # Array
    data = dataset["data"]
    print("Computing embeddings for the text in SQuAD")
    examples=[]
    for item in tqdm(data): 
        for para in item["paragraphs"]:
            
            # This is the short paragraph of context for the question
            context_tokens = para["context"]
            

            for qas in para["qas"]:

                # Question text
                question_tokens = qas["question"]

                # Unique identifier for (question, corresponding answers)
                qas_id = qas["id"]

                is_impossible = qas["is_impossible"]

                # Each question typically has multiple answers
                # Could be empty (if is_impossible = True, new in SQuAD v2)
                for ans in qas["answers"]:
                    
                    answer_text_tokens = ans["text"]
                    answer_start = ans["answer_start"]
                    example.append((context_tokens,question_tokens,answer_tokens))

                    #print(answer_start)
    with open('data/preprocessed_'+type+'_context.txt', 'w') as context_file,  \
         open('data/preprocessed_'+type+'_question.txt', 'w') as question_file,\
         open('data/preprocessed_'+type+'_ans_text.txt', 'w') as ans_text_file :
    
    # Write to files 
      for (context, question, answer) in examples:
          context_file.write(context_tokens.encode('utf8') + '\n')
          question_file.write(question_tokens.encode('utf8') + '\n')
          ans_text_file.write(answer_text_tokens.encode('utf8') + '\n')


def main():
    
    
    # download train set
     train_data = load_tokenize_train_set()

     # preprocess train set and write to file
     preprocess(train_data, 'training')

     # download dev set
     dev_data= load_tokenize_dev_set()

     # preprocess dev set and write to file
     preprocess(dev_data, 'developing')


if __name__ == '__main__':
      main()
