
import torch
import torch.nn as nn
import torch.nn.functional as F
def get_loss_function(loss_type,full_package):
    # Note: we can use information from full_package to customize the loss function
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_type == 'bce':
        loss_fn = nn.BCELoss()
    elif loss_type == 'pls':
        loss_fn = pls()
    elif loss_type == 'nls':
        loss_fn = nls()
    elif loss_type == 'fdiv':
        loss_fn = fdiv()
    elif loss_type == 'drops':
        loss_fn = nn.CrossEntropyLoss() 
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    full_package['criterion'] = loss_fn
    return full_package
    


class pls(nn.Module):
    def __init__(self, smooth_rate=0.6):
        super(pls, self).__init__()
        self.smooth_rate = smooth_rate
        self.confidence = 1 - self.smooth_rate
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y, t):
        assert y.shape[0] == t.shape[0], 'y and t must have the same batch size'
        assert len(y.shape) == 2 and len(t.shape) == 1, 'y must be 2D and t must be 1D'

        loss = self.cross_entropy(y, t)
        loss_ = -torch.log(F.softmax(y,dim=1) + 1e-8)
        loss = self.confidence * loss + self.smooth_rate * torch.mean(loss_, 1)
        return torch.sum(loss) / y.shape[0]


class nls(nn.Module):
    def __init__(self, smooth_rate=0.6):
        super(nls, self).__init__()
        self.smooth_rate = smooth_rate
        self.confidence = 1 - self.smooth_rate
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y, t):
        assert y.shape[0] == t.shape[0], 'y and t must have the same batch size'
        assert len(y.shape) == 2 and len(t.shape) == 1, 'y must be 2D and t must be 1D'

        loss = self.cross_entropy(y, t)
        loss_ = -torch.log(F.softmax(y,dim=1) + 1e-8)



        loss_nls = self.confidence * loss + self.smooth_rate * torch.mean(loss_, 1)
        return torch.sum(loss_nls) / y.shape[0]

class fdiv(nn.Module):
    def __init__(self):
        super(fdiv, self).__init__()
        self.nill_loss = nn.NLLLoss(reduction='none')


    def activation(self, x): return -torch.mean(torch.tanh(x) / 2.)
  
    def conjugate(self, x): return -torch.mean(torch.tanh(x) / 2.)

    def forward(self, y, y_peer, t, t_peer):
        assert y.shape[0] == t.shape[0] and y_peer.shape[0] == t_peer.shape[0], 'y and t must have the same batch size'
        assert len(y.shape) == 2 and len(t.shape) == 1 and len(y_peer.shape) == 2 and len(t_peer.shape) == 1, 'y must be 2D and t must be 1D'

        prob_acti = self.nill_loss(y, t)
        prob_conj = self.nill_loss(y_peer, t_peer)
        loss = self.activation(prob_acti) - self.conjugate(prob_conj)
        
        return torch.sum(loss)/y.shape[0]
    

if __name__ == '__main__':
    loss_names = ['ce','pls', 'nls', 'fdiv']
    logit = torch.randn(32, 10)
    label = torch.randint(0, 10, (32,))
    for loss_name in loss_names:
        criterion = get_loss_function(loss_name)
        try:
            loss = criterion(logit, label)
        except:
            loss = criterion(logit, logit, label, label)
        print(f'{loss_name}: {loss}')
    pass
