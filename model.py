import torch.nn as nn
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup, BertForSequenceClassification

class MyBertModel(nn.Module):
    def __init__(self):
        super(MyBertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = bert_out.logits
        return out


