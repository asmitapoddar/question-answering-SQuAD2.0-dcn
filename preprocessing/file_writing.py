from tqdm import tqdm 
import json
## Creates files for each group of data


def load_tokenize_train_set(): 
    with open("data/train-v2.0-tokenized.json", "r") as f:
        train_set = json.load(f)
    return train_set

def load_tokenize_dev_set(): 
    with open("data/dev-v2.0-tokenized.json", "r") as f:
        dev_set = json.load(f)
    return dev_set

def get_token_index(char_index, tokens):
    """
    Given a char index, returns corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        for 0,1,2,3,4 we return 0 and 6,7,8,9,10 we return 1.
    """
    acc = 0 # accumulator
    token_index=None
    current_token_index = 0 # current token location

    for current_token in tokens:
      for current_char in current_token:
        if acc==char_index:
          token_index=current_token_index
        acc+=1
      current_token_index+=1

    return token_index

def get_char_length(tokens):
  len=0
  for token in tokens:
    for char in token:
      len+=1
  return len

def preprocess(dataset, type):
    # Takes the parsed JSON dataset, and replaces the questions and context documents 
    # by a sequence of word embeddings after tokenisation.

    # Array
    data = dataset["data"]
    print("Computing embeddings for the text in SQuAD")
    examples=[]
#    first=True
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
                    answer_end=answer_start+get_char_length(answer_text_tokens) ####get the right len
                    assert answer_start <= answer_end
                    answer_token_start_location=get_token_index(answer_start, context_tokens)
                    answer_token_end_location=get_token_index(answer_end, context_tokens)
                   # if type=='developing' and first:
                    
                    #  if answer_token_start_location==None:
                     #   first=False
                      #  print("answer_token_start_location=None")
                      #  print(answer_start)
                      #  print("context length")
                      #  print(get_char_length(context_tokens))
                      #  print(answer_text_tokens)
                    

                    #if answer_token_end_location==None:
                     # print("answer_token_end_location=None")
                      #print(answer_end)
                    examples.append((' '.join(context_tokens), \
                     ' '.join(question_tokens), \
                     ' '.join(answer_text_tokens),\
                     ' '.join([str(answer_token_start_location), str(answer_token_end_location)])))
               
                    #print(answer_start)
    with open('data/preprocessed_'+type+'_context.txt', 'wb+') as context_file,  \
         open('data/preprocessed_'+type+'_question.txt', 'wb+') as question_file,\
         open('data/preprocessed_'+type+'_ans_text.txt', 'wb+') as ans_text_file,\
         open('data/preprocessed_'+type+'_ans_span.txt', 'wb+') as ans_span_file :

    
    # Write to files 
      for (context, question, answer, span) in examples:
          context_file.write(context.encode('utf8') + b'\n')
          question_file.write(question.encode('utf8') + b'\n')
          ans_text_file.write(answer.encode('utf8') + b'\n')
          ans_span_file.write(span.encode('utf8') + b'\n')



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
