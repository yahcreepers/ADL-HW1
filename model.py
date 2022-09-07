from typing import Dict

import torch
from torch.nn import Embedding
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
from torch.nn import functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, bid, output_size):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bid, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.act = nn.Sigmoid()

    def forward(self, x, text_lengths, c):
        packed_embedded = pack_padded_sequence(x, text_lengths,batch_first=True,enforce_sorted=False)
        out, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        outputs=self.act(dense_outputs)
        return outputs
        
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, bid, output_size):
        super(RNNModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bid, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, text_lengths, c):
        packed_embedded = pack_padded_sequence(x, text_lengths,batch_first=True)
        out, hn = self.rnn(packed_embedded)
        out = pad_packed_sequence(out, batch_first=True)
        hid = torch.stack([out[0][i][int(out[1][i]) - 1] for i in range(len(out[0]))])
        out = self.fc(hid)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, bid, output_size):
        super(GRUModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bid, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, text_lengths):
        packed_embedded = pack_padded_sequence(x, text_lengths,batch_first=True)
        out, hn = self.gru(packed_embedded)
        out = pad_packed_sequence(out, batch_first=True)
        hid = torch.stack([out[0][i][int(out[1][i]) - 1] for i in range(len(out[0]))])
        out = self.fc(hid)
        return out

class SLOT_GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, bid, output_size):
        super(SLOT_GRUModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bid, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, text_lengths, c):
        packed_embedded = pack_padded_sequence(x, text_lengths,batch_first=True)
        #h_0 = torch.rand(2*self.n_layers, len(text_lengths), self.hidden_size).cuda(c)
        out, hn = self.gru(packed_embedded)
        #print(t)
        out = pad_packed_sequence(out, batch_first=True)
        out = pad_sequence([self.fc(out[0][i][:text_lengths[i]]) for i in range(len(text_lengths))], batch_first=True)
        #print(out)
        #out = pack_padded_sequence(out, text_lengths,batch_first=True)
        #out = pad_packed_sequence(out, batch_first=True)
        #print(out)
        #return out[0]
        return out

class GRU_ATTModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, bid, output_size):
        super(GRU_ATTModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bid, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
    
    def attention_layer(self, output, h_t):
        batch_size = len(output)
        h = torch.cat((h_t[-2],h_t[-1]),dim=1).unsqueeze(2)
        attn_weights = torch.bmm(output, h).squeeze(2)
        attn_weights = F.softmax(attn_weights,1)
        context = torch.bmm(output.transpose(1,2),attn_weights.unsqueeze(2)).squeeze(2)
        return context
    
    def forward(self, x, text_lengths, c):
        packed_embedded = pack_padded_sequence(x, text_lengths,batch_first=True)
        #h_0 = torch.rand(2*self.n_layers, len(text_lengths), self.hidden_size).cuda(c)
        out, hn = self.gru(packed_embedded)
        out = pad_packed_sequence(out, batch_first=True)
        #hid = torch.stack([out[0][i][int(out[1][i]) - 1] for i in range(len(out[0]))])
        #print(out[0].shape, hn.shape)
        hid = self.attention_layer(out[0], hn)
        #print(hid.shape)
        out = self.fc(hid)
        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
