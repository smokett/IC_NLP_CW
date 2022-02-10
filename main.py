import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
from data import dataset
from model import MyBertModel
from trainer import Trainer
from utils import get_df
from data_analysis import Preprocessor

# Check if using cuda
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

# Load data
path = 'nlp_data'
df_train, df_test, _, _ = get_df(path)

# Preprocessing



# Define tokenizer
tk = AutoTokenizer.from_pretrained("roberta-base")

# Prepare dataset
train_data = dataset(df_train,tk)
val_data = dataset(df_test,tk)

# Rebalance data if necessary
train_sample_weights = train_data.get_sample_weights()
train_weighted_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_data), replacement=True)

# Prepare dataloader
train_dataloader = DataLoader(dataset = train_data, batch_size=8, sampler=train_weighted_sampler)
val_dataloader = DataLoader(dataset = val_data, batch_size=8, shuffle=False)

# Define our Trainer class
trainer = Trainer(MyBertModel(), train_dataloader, val_dataloader)
# -- Start Training -- #
trainer.train()