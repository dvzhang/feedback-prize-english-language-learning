import pandas as pd
import numpy as np
import os

def getfiles():
    filenames=os.listdir(r'input/feedback-prize-2021/train')
    return ['input/feedback-prize-2021/train/' + file for file in filenames]

train = pd.read_csv('./input/feedback-prize-english-language-learning/train.csv')
test = pd.read_csv('./input/feedback-prize-english-language-learning/test.csv')
# lang = pd.read_csv('data/static/lang8.csv')

res = []
for i in getfiles():
    with open(i, "r") as f:
        lines = f.readlines()
    res.append(" ".join(lines).strip())

with open(i, "r") as f:
    lines = f.readlines()
for line in lines:
    res.append(line.strip())

# res = pd.DataFrame(res)

print(train.head())
mlm_data = train[['full_text']]
mlm_data = mlm_data.rename(columns={'full_text':'text'})
for line in res:
    mlm_data.append({'text':line.replace("\n", " ")}, ignore_index=True)


mlm_data.to_csv('./input/mlm_data.csv', index=False)



