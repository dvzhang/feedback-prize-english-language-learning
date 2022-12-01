import numpy as np 
import pandas as pd 
import os, gc, re, warnings
from sklearn.metrics import mean_squared_error

from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train_df = pd.read_csv("./input/feedback-prize-english-language-learning/train.csv")
train_df["type"]="train"

test_df = pd.read_csv("./input/feedback-prize-english-language-learning/test.csv")
test_df["type"]="test"
all_df = pd.concat([train_df,test_df],ignore_index=True)
TARGET = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]
print(all_df.head())


import sys
sys.path.append('./input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
FOLDS = 5
skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
for i,(train_index, val_index) in enumerate(skf.split(train_df,train_df[TARGET])):
    train_df.loc[val_index,'FOLD'] = i


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


BATCH_SIZE = 4

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"full_text"]
        tokens = tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN,return_tensors="pt")
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return tokens

ds_tr = EmbedDataset(train_df)
embed_dataloader_tr = torch.utils.data.DataLoader(ds_tr,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)
ds_te = EmbedDataset(test_df)
embed_dataloader_te = torch.utils.data.DataLoader(ds_te,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)

tokenizer = None
MAX_LEN = 640

def get_embeddings(MODEL_NM='', MAX=250, BATCH_SIZE=4, verbose=True):
    global tokenizer, MAX_LEN
    DEVICE="cuda"
    model = AutoModel.from_pretrained( MODEL_NM )
    # tokenizer = AutoTokenizer.from_pretrained( MODEL_NM )
    tokenizer = AutoTokenizer.from_pretrained( 'roberta-base' )
    MAX_LEN = MAX
    
    model = model.to(DEVICE)
    model.eval()
    all_train_text_feats = []
    for batch in tqdm(embed_dataloader_tr,total=len(embed_dataloader_tr)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        all_train_text_feats.extend(sentence_embeddings)
    all_train_text_feats = np.array(all_train_text_feats)
    if verbose:
        print('Train embeddings shape',all_train_text_feats.shape)
        
    te_text_feats = []
    for batch in tqdm(embed_dataloader_te,total=len(embed_dataloader_te)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        te_text_feats.extend(sentence_embeddings)
    te_text_feats = np.array(te_text_feats)
    if verbose:
        print('Test embeddings shape',te_text_feats.shape)
        
    return all_train_text_feats, te_text_feats


# MODEL_NM = './input/huggingface-deberta-variants/deberta-base/deberta-base'
MODEL_NM = './output'
all_train_text_feats, te_text_feats = get_embeddings(MODEL_NM)


def GetBasedModel():
    basedModels = []
    basedModels.append(('SVR'  , SVR(C=1,kernel='poly',epsilon=0.2,gamma='scale')))
    basedModels.append(('RGE'  , Ridge(alpha=0.1,solver='saga')))
    return basedModels
models = GetBasedModel()

# FOLDS = 10
# def comp_score(y_true,y_pred):
#     rmse_scores = []
#     for i in range(len(TARGET)):
#         rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
#     return np.mean(rmse_scores)

# param_grid_RGE = {
#     'solver': ['auto', 'svd', 'cholesky','lsqr', 'sparse_cg','sag', 'saga', 'lbfgs'],
#     'alpha': [0.1,0.2,0.3]
#     }

# param_grid_SVR = {
#     'C': [0.1,0.5,1.0],
#     'kernel': ['linear','poly','rbf','sigmoid'],
#     'gamma': ['scale', 'auto'],
#     'epsilon': [0.1,0.2,0.3]
#     }

# for name, clf in models:
#     print(name)
#     preds = []
#     scores = []
    
#     for fold in range(FOLDS):
#         print('#'*10)
#         print('### Fold',fold+1)
#         print('#'*10)

#         train_df_ = train_df[train_df["FOLD"]!=fold]
#         val_df_ = train_df[train_df["FOLD"]==fold]

#         tr_text_feats = all_train_text_feats[list(train_df_.index),:]
#         ev_text_feats = all_train_text_feats[list(val_df_.index),:]

#         ev_preds = np.zeros((len(ev_text_feats),6))
#         test_preds = np.zeros((len(te_text_feats),6))
#         for i,t in enumerate(TARGET):
#             print(t,', ',end='')
#             CV = GridSearchCV(clf,param_grid_SVR, cv=3, n_jobs= 1)
#             CV.fit(tr_text_feats,train_df_[t].values)  
#             print(CV.best_params_)    
#             print(CV.best_score_)
            
            
#             #clf.fit(tr_text_feats, train_df_[t].values)
#             #ev_preds[:,i] = clf.predict(ev_text_feats)
#             #test_preds[:,i] = clf.predict(te_text_feats)
#         #print()
#         #score = comp_score(val_df_[TARGET].values,ev_preds)
#         #scores.append(score)
#         #print("Fold : {} RSME score: {}".format(fold,score))
#         #preds.append(test_preds)

#     #print('#'*10)
#     print('Overall CV RSME =',np.mean(scores))


FOLDS = 5
def comp_score(y_true,y_pred):
    rmse_scores = []
    for i in range(len(TARGET)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)

for name, clf in models:
    print(name)
    preds = []
    scores = []
    
    for fold in range(FOLDS):
        print('#'*10)
        print('### Fold',fold+1)
        print('#'*10)

        train_df_ = train_df[train_df["FOLD"]!=fold]
        val_df_ = train_df[train_df["FOLD"]==fold]

        tr_text_feats = all_train_text_feats[list(train_df_.index),:]
        ev_text_feats = all_train_text_feats[list(val_df_.index),:]

        ev_preds = np.zeros((len(ev_text_feats),6))
        test_preds = np.zeros((len(te_text_feats),6))
        for i,t in enumerate(TARGET):
            print(t,', ',end='')           
            clf.fit(tr_text_feats, train_df_[t].values)
            ev_preds[:,i] = clf.predict(ev_text_feats)
            test_preds[:,i] = clf.predict(te_text_feats)
        print()
        score = comp_score(val_df_[TARGET].values,ev_preds)
        scores.append(score)
        print("Fold : {} RSME score: {}".format(fold+1,score))
        preds.append(test_preds)

    print('#'*10)
    print('Overall CV RSME =',np.mean(scores))
    
    sub = test_df.copy()
    sub.loc[:,TARGET] = np.average(np.array(preds),axis=0) #,weights=[1/s for s in scores]
    sub_columns = pd.read_csv("./input/feedback-prize-english-language-learning/sample_submission.csv").columns
    sub = sub[sub_columns]
    sub.to_csv(f"submission_{name}.csv",index=None)
    sub.head()

pred_SVR = pd.read_csv('submission_SVR.csv')
pred_RGE = pd.read_csv('submission_RGE.csv')

submission_file = pd.read_csv("./input/feedback-prize-english-language-learning/sample_submission.csv")
submission_file.loc[:,TARGET] = (pred_SVR.loc[:,TARGET] + pred_RGE.loc[:,TARGET])/2

submission_file.to_csv(f"submission.csv",index=None)
submission_file.to_csv(f"submission_V2.csv",index=None)
print(submission_file.head())