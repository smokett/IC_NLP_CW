import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from data import dataset
from model import MyBertModel
from trainer import Trainer
from utils import get_df
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

path = 'nlp_data'
df_train, df_test, _, _ = get_df(path)
tk = AutoTokenizer.from_pretrained("roberta-base")

train_data = dataset(df_train,tk)
val_data = dataset(df_test,tk)

train_sample_weights = train_data.get_sample_weights()
train_weighted_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_data), replacement=True)

train_dataloader = DataLoader(dataset = train_data, batch_size=8, shuffle=True, sampler=train_weighted_sampler)
val_dataloader = DataLoader(dataset = val_data, batch_size=8, shuffle=False)
trainer = Trainer(MyBertModel(), train_dataloader, val_dataloader)
trainer.train()