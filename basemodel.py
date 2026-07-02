import torch
import torch.nn
import pyexp
from . import DictDataset

class BaseModel(torch.nn.Module):


    def __init__(self, dict_dataset: DictDataset, device='cpu',):
        super().__init__()
        # select device
        self.device = torch.device(device)
        self.to(self.device)
        # load dataset
        self.dict_dataset:DictDataset = dict_dataset
        for name_var in ['train_x', 'train_y', 'valid_x', 'valid_y']:
            if self.dict_dataset[name_var] is not None:
                self.dict_dataset[name_var]= self.dict_dataset[name_var].to(device)


    def train_model(
            self,
            learn_rate=0.03,
            num_epoch=200, ):

        train_x = self.dict_dataset['train_x']
        train_y = self.dict_dataset['train_y']
        valid_x = self.dict_dataset.get('valid_x', None)
        valid_y = self.dict_dataset.get('valid_y', None)

        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)
        criterion = torch.nn.MSELoss(reduction='mean')

        list_loss_train = []
        list_loss_valid = []

        for ii in range(num_epoch):

            # train
            self.train()
            optimizer.zero_grad()
            train_y_pred = self.forward(train_x)
            loss_train = criterion(train_y_pred, train_y)
            loss_train.backward()
            optimizer.step()
            loss_train = loss_train.item()
            # valid
            loss_valid = 0.0
            if valid_x is not None:
                self.eval()
                with torch.no_grad():
                    valid_y_pred = self.forward(valid_x)
                    loss_valid = criterion(valid_y_pred, valid_y)
                    loss_valid = loss_valid.item()
            # print
            str_end = '\n' if ii == num_epoch - 1 else '\r'
            print(
                '(%4d/%4d) train: %10.4e, valid: %10.4e'
                % (ii + 1, num_epoch, loss_train, loss_valid),
                end=str_end)
            # save
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)
        
        # update pred in dict
        self.dict_dataset['train_y_pred'] = train_y_pred
        self.dict_dataset['valid_y_pred'] = valid_y_pred

        return list_loss_train, list_loss_valid
    
    
    def ploy_pairty(self):

        fig, ax = pyexp.initial_fig_ax()
        train_y = self.dict_dataset['train_y'].detach().cpu().numpy()
        valid_y = self.dict_dataset['valid_y'].detach().cpu().numpy()
        train_y_pred = self.dict_dataset['train_y_pred'].detach().cpu().numpy()
        valid_y_pred = self.dict_dataset['valid_y_pred'].detach().cpu().numpy()
        ax.scatter(train_y, train_y_pred, 10, color='C0', label='train', zorder=1)
        ax.scatter(valid_y, valid_y_pred, 10, color='C1', label='valid', zorder=1)
        ax.set_aspect('equal')
        ax.set_xlabel('Raw', fontsize=pyexp.dict_conf_fontsize['label'])
        ax.set_ylabel('Predict', fontsize=pyexp.dict_conf_fontsize['label'])
        ax.plot(ax.get_xlim(), ax.get_xlim(), 'k-', alpha=0.5, zorder=0, scalex=False, scaley=False,)
        ax.legend(frameon=False, handletextpad=0.2, fontsize=pyexp.dict_conf_fontsize['legend'])