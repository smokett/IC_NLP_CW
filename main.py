import torch
import numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
from data import dataset
from model import MyBertModel
from trainer import Trainer
from utils import get_df, get_ext_df
from data_analysis import Preprocessor
from transformers import RobertaForSequenceClassification, BertForSequenceClassification

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(1)

# Check if using cuda
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

# Load data
path = 'nlp_data'
df_train, df_test, _, _ = get_df(path)
df_train_ext = get_ext_df(path)

# Useful settings (hyperparameters)
config = {
    'back_translation': True,
    'lr': 4e-5,
    'epochs': 20,
    'gradient_accumulate_steps': 2,
    'mo': None,
    'resample_scale':2,
    'input_max_length': 512,
    'batch_size': 1
}

# Preprocessing
# TO-DO


# Define tokenizer/Bert variant
tk = AutoTokenizer.from_pretrained("roberta-base")
bert_variant = RobertaForSequenceClassification.from_pretrained('roberta-base')
# tk = AutoTokenizer.from_pretrained("bert-base-cased")
# bert_variant = BertForSequenceClassification.from_pretrained('bert-base-cased')


# Prepare dataset
if config['back_translation']:
    train_data = dataset(df_train_ext, tk)
else:
    train_data = dataset(df_train, tk)
val_data = dataset(df_test, tk)

# Rebalance data if necessary
train_sample_weights = train_data.get_sample_weights(scaling=config['resample_scale'])
train_weighted_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_data), replacement=True)

# Prepare dataloader
train_dataloader = DataLoader(dataset=train_data, batch_size=config['batch_size'], sampler=train_weighted_sampler)
val_dataloader = DataLoader(dataset=val_data, batch_size=config['batch_size'], shuffle=False)



# Define our Trainer class
trainer = Trainer(MyBertModel(bert_variant), config, train_dataloader, val_dataloader)

# If load from pretrained
# trainer.from_checkpoint(model_path='models/saved_model.pt')
# -- Start Training -- #
trainer.train()

test_sent = 'I am test'
data = tk(test_sent, truncation=True, padding='max_length', max_length=config['input_max_length'], return_tensors='pt')

result = trainer.inference(data)
print(result)

