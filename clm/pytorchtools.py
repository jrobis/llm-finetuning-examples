import numpy as np
import torch


algos = ['Adam', "Adadelta", "Adagrad", "AdamW", "SparseAdam", "Adamax", "ASGD", "LBFGS", "NAdam",
 "RAdam", "RMSProp", "Rprop", "SGD"]
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print 
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = -0.01
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_optimizer(algo, parameters, lr):
    if algo in algos:
        idx = algos.index(algo)
        print(idx)
        if idx==0:
            return torch.optim.Adam(parameters, lr=lr)
        if idx==1:
            return torch.optim.Adadelta(parameters, lr=lr)
        if idx==2:
            return torch.optim.Adagrad(parameters, lr=lr)
        if idx==3:
            return torch.optim.AdamW(parameters, lr=lr)
        if idx==4:
            return torch.optim.SparseAdam(parameters, lr=lr)
        if idx==5:
            return torch.optim.Adamax(parameters, lr=lr)
        if idx==6:
            return torch.optim.ASGD(parameters, lr=lr)
        if idx==7:
            return torch.optim.LBFGS(parameters, lr=lr)
        if idx==8:
            return torch.optim.NAdam(parameters, lr=lr)
        if idx==9:
            return torch.optim.RAdam(parameters, lr=lr)
        if idx==10:
            return torch.optim.RMSprop(parameters, lr=lr)
        if idx==11:
            return torch.optim.Rprop(parameters, lr=lr)
        if idx==12:
            return torch.optim.SGD(parameters, lr=lr)
    else:
        print("Choose one of the following optimizers: {}".format(algos))
