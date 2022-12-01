from tqdm import tqdm, trange
from IPython.display import clear_output
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from transformers import RobertaConfig
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import gc
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import os
from glob import glob
import pandas as pd
import numpy as np

train = pd.read_csv('./input/commonlitreadabilityprize/train.csv')
test = pd.read_csv('./input/commonlitreadabilityprize/test.csv')

# %matplotlib inline
gc.enable()


def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
    data = data.replace('\n', '')
    tok = tokenizer.encode_plus(
        data,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )
    curr_sent = {}
    padding_length = max_len - len(tok['input_ids'])
    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    return curr_sent


class DatasetRetriever(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.excerpts = self.data.excerpt.values.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if not self.is_test:
            excerpt, label = self.excerpts[item], self.targets[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer,
                self.max_len, self.is_test
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'token_type_ids': torch.tensor(features['token_type_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.double),
            }
        else:
            excerpt = self.excerpts[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer,
                self.max_len, self.is_test
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'token_type_ids': torch.tensor(features['token_type_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
            }


class CommonLitModel(nn.Module):
    def __init__(
        self,
        model_name,
        config,
        multisample_dropout=False,
        output_hidden_states=False
    ):
        super(CommonLitModel, self).__init__()
        self.config = config
        self.roberta = RobertaModel.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        self.regressor = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[1]
        sequence_output = self.layer_norm(sequence_output)

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.regressor(dropout(sequence_output))
            else:
                logits += self.regressor(dropout(sequence_output))

        logits /= len(self.dropouts)

        # calculate loss
        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(labels.dtype)
            loss = torch.sqrt(loss_fn(logits, labels.view(-1)))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


def make_model(model_name='roberta-base', num_labels=1):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained(model_name)
    config.update({'num_labels': num_labels})
    model = CommonLitModel(model_name, config=config)
    return model, tokenizer




def make_loader(
    data,
    tokenizer,
    max_len,
    batch_size,
):

    test_dataset = DatasetRetriever(data, tokenizer, max_len, is_test=True)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size // 2,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=0
    )

    return test_loader


class Evaluator:
    def __init__(self, model, scalar=None):
        self.model = model
        self.scalar = scalar

    def evaluate(self, data_loader, tokenizer):
        preds = []
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                input_ids, attention_mask, token_type_ids = batch_data['input_ids'], \
                    batch_data['attention_mask'], batch_data['token_type_ids']
                input_ids, attention_mask, token_type_ids = input_ids.cuda(), \
                    attention_mask.cuda(), token_type_ids.cuda()

                if self.scalar is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )

                logits = outputs[0].detach().cpu().numpy().squeeze().tolist()
                preds += logits
        return preds


def config(fold):
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    
    max_len = 250
    batch_size = 8

    model, tokenizer = make_model(
        model_name='./output/', 
        num_labels=1
    )
    model.load_state_dict(
        torch.load(f'./model{fold}.bin')
    )
    test_loader = make_loader(
        test, tokenizer, max_len=max_len,
        batch_size=batch_size
    )

    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(), 
            torch.cuda.get_device_name(0))
        )
        model = model.cuda() 
    else:
        raise ValueError('CPU training is not supported')

    # scaler = torch.cuda.amp.GradScaler()
    scaler = None
    return (
        model, tokenizer, 
        test_loader, scaler
    )


def run(fold=0):
    model, tokenizer, \
        test_loader, scaler = config(fold)
    
    import time

    evaluator = Evaluator(model, scaler)

    test_time_list = []

    torch.cuda.synchronize()
    tic1 = time.time()

    preds = evaluator.evaluate(test_loader, tokenizer)

    torch.cuda.synchronize()
    tic2 = time.time() 
    test_time_list.append(tic2 - tic1)
    
    del model, tokenizer, test_loader, scaler
    gc.collect()
    torch.cuda.empty_cache()
    
    return preds


pred_df = pd.DataFrame()
for fold in tqdm(range(5)):
    pred_df[f'fold{fold}'] = run(fold)

sub = pd.read_csv('./input/commonlitreadabilityprize/sample_submission.csv')
sub['target'] = pred_df.mean(axis=1).values.tolist()
sub.to_csv('submission.csv', index=False)

print(sub)