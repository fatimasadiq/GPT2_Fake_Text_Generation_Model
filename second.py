import pandas as pd
import sklearn
import os
import dataclasses
from dataclasses import dataclass

# setup imports to use the model to generat some text
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer

model = TFGPT2LMHeadModel.from_pretrained("./model", from_pt=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", skip_special_token=True)

#input_ids = tokenizer.encode("Hello", return_tensors='tf')
input_ids = tokenizer.encode("<|startoftext|>", return_tensors='tf')
print(input_ids)

#generate text from input text
generated_text_samples = model.generate(
    input_ids, 
    max_length=30,  
    num_return_sequences=10,
    no_repeat_ngram_size=2,
    repetition_penalty=1.5,
    top_p=0.95,
    temperature=1.0,
    do_sample=True,
    top_k=50,
    early_stopping=True
    
)

#Print output for each sequence generated above
for i, beam in enumerate(generated_text_samples):
  print("{}: {}".format(i,tokenizer.decode(beam, skip_special_tokens=True)))
  print()




