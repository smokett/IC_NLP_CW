import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

def cal_acc(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, axis=1) == y_true) / len(y_true)

class Trainer(object):
    def __init__(self, model, train_loader, val_loader):

        self.model = model.to(device)
        self.epochs = 20
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=0.003)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=self.epochs * len(train_loader))
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metric = cal_acc

    def from_checkpoint(self, model_path):
        if os.path.exists(model_path):
            print('-'*60)
            print('Loading pretrained model:{}...'.format(model_path))
            print('-'*60)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print('-'*60)
            print('Success!')
            print('-'*60)
        else:
            print('-'*60)
            print('Provided path {} not found!'.format(model_path))
            print('-'*60)
    
    def save_checkpoint(self, save_path):
        torch.save(model.state_dict(), model_path)
        print('-'*60)
        print('Model saved as path {}!'.format(save_path))
        print('-'*60)

    def run_one_epoch(self, loader, logging_freq=10, eval=False ):
        # Moving average statistics
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        batch_loss = []
        batch_accuracy = []
        
        data_iter = tqdm(enumerate(loader), total=len(loader), bar_format="{bar}{l_bar}{r_bar}")
        for step, batch in data_iter:
            input_ids, attention_mask, y_true = [x.to(device) for x in batch]
            y_pred = self.model(input_ids, attention_mask)

            if not eval:
                self.optimizer.zero_grad()

                loss = self.loss_fn(y_pred, y_true)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                accuracy = self.metric(y_pred, y_true)
                loss = loss.cpu().item() if use_cuda else loss.item()
                accuracy = accuracy.cpu().item() if use_cuda else accuracy.item()     
                
                batch_loss.append(loss)
                batch_accuracy.append(accuracy)

            else:
                loss = self.loss_fn(y_pred, y_true)
                accuracy = self.metric(y_pred, y_true)

                loss = loss.cpu().item() if use_cuda else loss.item()
                accuracy = accuracy.cpu().item() if use_cuda else accuracy.item()  

                batch_loss.append(loss)
                batch_accuracy.append(accuracy)
                
            epoch_loss = np.sum(batch_loss)
            epoch_accuracy += np.sum(batch_accuracy)
            mode = "Train" if not eval else "Eval"
            if (step+1) % logging_freq == 0: # Use 1-based index for logging
                data_iter.set_description("Mode: {} | Step: {} | Loss: {} | Accuracy: {}".format(
                    mode, step + 1, 
                    np.mean(batch_loss[(step + 1 - logging_freq) : (step+1)]), 
                    np.mean(batch_accuracy[(step + 1 - logging_freq) : (step+1)])
                ))
            if (step) == len(loader)-1:
                data_iter.set_description("Mode: {} | End of Epoch | Loss: {} | Accuracy: {}".format(
                    mode, 
                    epoch_loss/(step+1),
                    epoch_accuracy/(step+1)
                ))
            
        return epoch_loss, accuracy
    
    def train(self, val_freq=20):
        # Moving average statistics
        train_loss = 0.0
        train_accuracy = 0.0
        val_loss = 0.0
        val_accuracy = 0.0

        for i in range(self.epochs):
            self.model.train()
            print('-' * 30 + 'Train for Epoch {}'.format(i) + '-'*30 )
            epoch_loss, epoch_accuracy = self.run_one_epoch(self.train_loader, logging_freq=10, eval=False)
            
            train_loss += epoch_loss
            train_accuracy += epoch_accuracy

            print("Mode: Train | Epoch: {} | Loss: {} | Accuracy: {}".format(
                  i + 1, train_loss / (i+1), train_accuracy / (i+1)
            ))
            if i % val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    print('-' * 30 + 'Val at Epoch{}'.format(i) + '-'*30 )
                    epoch_loss, epoch_accuracy = self.run_one_epoch(self.val_loader, logging_freq=10, eval=True)

                    val_loss += epoch_loss
                    val_accuracy += epoch_accuracy
                    print("Mode: Eval | Epoch: {} | Loss: {} | Accuracy: {}".format(
                        i + 1, val_loss / (i+1), val_accuracy / (i+1)
                    ))

            