import torch
from torch.utils.data import Dataset

## This file contains Dataset class
class dataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tk = tokenizer

        self.data_encoded = [self.tk(sent, truncation=True, padding='max_length', max_length=512, return_tensors='pt') for sent in self.df['paragraph']]
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        data_encoded = self.data_encoded[index]
        label = torch.LongTensor([self.df['label'].iloc[index]]).squeeze()
        return data_encoded['input_ids'].squeeze(), data_encoded['attention_mask'].squeeze(), label

    def get_sample_weights(self, scaling=1):
        weights = self.df['label'].value_counts().min() / self.df['label'].value_counts()[self.df['label']]
        weights = weights.to_list()
        weights = [w*scaling if w!=1 else w for w in weights]
        return weights


if __name__=='__main__':
    from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
    from torch.utils.data import DataLoader
    from utils import get_df

    path = 'nlp_data'
    tk = AutoTokenizer.from_pretrained("bert-base-cased")
    df_train, df_test, _, _ = get_df(path)
    train_data = dataset(df_train,tk)
    val_data = dataset(df_test,tk)
    train_dataloader = DataLoader(dataset = train_data, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(dataset = val_data, batch_size=8, shuffle=False)
    
    a,b,c = next(iter(train_dataloader))
    print(a.shape, b.shape, c.shape)