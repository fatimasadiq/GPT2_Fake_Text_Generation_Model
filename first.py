import pandas as pd
import sklearn
import os
import dataclasses
from dataclasses import dataclass

df = pd.read_csv('input.txt',delimiter='\t' , index_col= False , usecols=["TITLE"])
# df.head()

df.to_csv (r'./newdata.txt', index = False, header=True)

"""

splitting data into training and evaluation set
Now load the data line by line
"""

from sklearn.model_selection import train_test_split

with open('newdata.txt', 'r') as data:
  dataset = ["<|startoftext|>" + x.strip() + "<|endoftext|>" for x in data.readlines()]


train, eval = train_test_split(dataset, train_size=.9, random_state=2020)
print("training size:" + str(len(train)))
print("Evaluation size: " + str(len(eval)))

with open('train_tmp.txt', 'w') as file_handle:
  file_handle.write("\n".join(train))
  

with open('eval_tmp.txt', 'w') as file_handle:
  file_handle.write("\n".join(eval))


