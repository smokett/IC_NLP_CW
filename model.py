import torch.nn as nn

class MyBertModel(nn.Module):
    def __init__(self, bert_variant):
        super(MyBertModel, self).__init__()
        self.bert = bert_variant
        
    def forward(self, inputs):
        input_ids = inputs[0]
        attention_mask = inputs[1]
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = bert_out.logits
        return out

class MyBertModel_2(nn.Module):
    def __init__(self, bert_variant):
        super(MyBertModel_2, self).__init__()
        self.bert = bert_variant
        self.embedding = nn.Embedding(10, 100)
        self.bert_out_proj = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear(768, 100, bias=True)
            )
        self.final_out_proj = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear(100, 2, bias=True)
            )
        # self.bert.classifier = self.classifier
    def forward(self, inputs):
        input_ids = inputs[0]
        attention_mask = inputs[1]
        keyword = inputs[2]
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        bert_proj = self.bert_out_proj(bert_out)
        emb_out = self.embedding(keyword)
        out = self.final_out_proj(emb_out+bert_proj)

        return out
if __name__ == '__main__':
    from transformers import RobertaForSequenceClassification, BertForSequenceClassification, RobertaModel, RobertaConfig
    configuration = RobertaConfig()
    # Initializing a model from the configuration
    model = RobertaModel(configuration, add_pooling_layer = False)
    bert_variant = RobertaModel.from_pretrained('roberta-base',add_pooling_layer = False, attention_probs_dropout_prob=0.2,hidden_dropout_prob=0.2)
    mybert = MyBertModel_2(bert_variant)
    print(mybert.bert)