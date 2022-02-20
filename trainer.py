import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

def cal_acc(y_pred, y_true):
    """
    Calculate accuracy
    """
    return torch.sum(y_pred.argmax(dim=1) == y_true) / len(y_true)

def f1_loss(y_pred, y_true, is_training=False):
    """
    Calculate F1-score
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    tp = (y_true * y_pred).sum().to(torch.float32) 
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32) 
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32) 
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32) 
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

class Trainer(object):
    """
    Our trainer of BERT model.
    """
    def __init__(self, model, config, train_loader, val_loader):

        self.config = config
        self.epochs = self.config['epochs']
        self.gas = self.config['gradient_accumulate_steps']
        self.lr = self.config['lr']

        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0, # Default value
            num_training_steps=self.epochs * (len(train_loader)//self.gas) # Note the traing steps also adjust based on gas
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metric = cal_acc



    def from_checkpoint(self, model_path):
        """
        Function to load trained model
        """
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

    def run_one_epoch(self, loader, logging_freq, eval=False ):
        """
        Fuction to train for one epoch
        loader: dataloader
        logging_freq: how many steps do we log the running stat
        eval: whether running train or evaluation
        """
        # Moving average statistics
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        batch_loss = []
        batch_accuracy = []

        all_y_true = []
        all_y_pred = []
        
        data_iter = tqdm(enumerate(loader), total=len(loader), bar_format="{bar}{l_bar}{r_bar}")
        for step, batch in data_iter:
            input_ids, attention_mask, y_true = [x.to(device) for x in batch]
            y_pred = self.model(input_ids, attention_mask)

            # Since F1 only calculated per whole Epoch
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            # Training, actively update parameters
            if not eval:
                self.optimizer.zero_grad()

                # Normalise the loss if accumulation applied (decouple from learning rate)
                loss = self.loss_fn(y_pred, y_true)/self.gas
                loss.backward()

                # Accumulate gradient to effectively increase batch size
                if step % self.gas == 0:
                    self.optimizer.step()
                    self.scheduler.step()

                accuracy = self.metric(y_pred, y_true)
                loss = loss.cpu().item()
                accuracy = accuracy.cpu().item()    
                
                batch_loss.append(loss)
                batch_accuracy.append(accuracy)

            # Evaluation, no gradient
            else:
                loss = self.loss_fn(y_pred, y_true)
                accuracy = self.metric(y_pred, y_true)

                loss = loss.cpu().item()
                accuracy = accuracy.cpu().item()

                batch_loss.append(loss)
                batch_accuracy.append(accuracy)
                

            # Logging statistics
            mode = "Train" if not eval else "Eval"
            # Moving average of batches
            if (step+1) % logging_freq == 0: # Use 1-based index for logging
                data_iter.set_description("[Batch {} Running Stat] Mode: {} | Step: {} | Loss: {} | Metric: {}".format(
                    logging_freq, mode, step + 1,
                    np.mean(batch_loss[(step + 1 - logging_freq) : (step+1)]), 
                    np.mean(batch_accuracy[(step + 1 - logging_freq) : (step+1)])
                )
            )
            # Whole average of batch in one epoch
            if (step) == len(loader)-1:
                data_iter.set_description("[Per Batch Stat] Mode: {} | End of Epoch | Loss: {} | Metric: {}".format(
                    mode, 
                    np.mean(batch_loss),
                    np.mean(batch_accuracy)
                )
            )
        epoch_f1 = f1_loss(torch.cat(all_y_pred), torch.cat(all_y_true))
        epoch_loss = np.mean(batch_loss)
        epoch_accuracy = np.mean(batch_accuracy)
            
        return epoch_loss, epoch_accuracy
    
    def train(self, logging_freq=10, val_freq=20):
        """
        Function to train the model
        logging_freq: how many steps do we log the running stat
        val_freq: frequency to do evaluation
        """
        # Moving average statistics
        train_loss = 0.0
        train_accuracy = 0.0
        val_loss = 0.0
        val_accuracy = 0.0

        for i in range(self.epochs):
            self.model.train()
            print('-' * 30 + 'Train for Epoch {}'.format(i+1) + '-'*30 )
            epoch_loss, epoch_accuracy = self.run_one_epoch(
                self.train_loader, 
                logging_freq=logging_freq, 
                eval=False
            )
            
            train_loss += epoch_loss
            train_accuracy += epoch_accuracy

            print("[Epoch Running Stat] Mode: Train | Epoch: {} | Loss: {} | Metric: {}".format(
                  i + 1, train_loss / (i+1), train_accuracy / (i+1)
                )
            )

            # Do evaluation 
            if i % val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    print('-' * 30 + 'Val at Epoch{}'.format(i+1) + '-'*30 )
                    epoch_loss, epoch_accuracy = self.run_one_epoch(
                        self.val_loader, 
                        logging_freq=logging_freq, 
                        eval=True
                    )

                    val_loss += epoch_loss
                    val_accuracy += epoch_accuracy
                    print("[Epoch Running Stat] Mode: Eval | Epoch: {} | Loss: {} | Metric: {}".format(
                        i//val_freq + 1, val_loss / (i//val_freq+1), val_accuracy / (i//val_freq+1)
                    ))
    def inference(self, data):
        with torch.no_grad():
            data = data.to(device)
            y_pred = self.model(data['input_ids'], data['attention_mask'])
            y_pred = y_pred.argmax(dim=1)
            return y_pred.cpu().item()

            