import pandas as pd
import numpy as np

train = pd.read_csv('../input/commonlitreadabilityprize/train.csv')
test = pd.read_csv('../input/commonlitreadabilityprize/test.csv')

mlm_data = train[['excerpt']]
mlm_data = mlm_data.rename(columns={'excerpt':'text'})
mlm_data.to_csv('mlm_data.csv', index=False)

mlm_data_val = test[['excerpt']]
mlm_data_val = mlm_data_val.rename(columns={'excerpt':'text'})
mlm_data_val.to_csv('mlm_data_val.csv', index=False)