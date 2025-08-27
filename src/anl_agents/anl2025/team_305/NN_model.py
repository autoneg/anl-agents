import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyLSTMpro(nn.Module):
    def __init__(self):
        super(MyLSTMpro,self).__init__()
        self.flag=nn.Embedding(3,4)
        self.cent=nn.Embedding(2,4)
        self.lstm=nn.LSTM(10,128,batch_first=True)
        self.fc=nn.Linear(128,1)
        self.pre_fc = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU()
        )
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.5)

    
    def forward(self,x,hidden=None):
        flag=x[:,:,0].long()
        ufun=x[:,:,1].unsqueeze(-1)
        time=x[:,:,2].unsqueeze(-1)
        cent=x[:,:,3].long()
        centemb=self.cent(cent)
        flagemb=self.flag(flag)
        n=torch.cat([flagemb,ufun,time,centemb],dim=-1)

        lengths= (flag!=0).sum(dim=1)
        packed = pack_padded_sequence(n, lengths, batch_first=True, enforce_sorted=False)
        lstm_out_packed, hidden = self.lstm(packed, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        final_out = torch.stack([lstm_out[i, lengths[i]-1] for i in range(len(lengths))])
        #for i_index,row in enumerate(x):
        #    for j_index,value in enumerate(row):
        #        flag=value[0].long()
        #        ufun=value[1].unsqueeze(-1)
        #        flagemb=self.flag(flag)
        #        n[i_index][j_index]=torch.cat((flagemb,ufun),dim=-1)
        
        
        #lstm_out, hidden=self.lstm(n,hidden)
        #out=self.pre_fc(lstm_out[:,-1])


        out=self.pre_fc(final_out)
        out=self.fc(out)
        out=torch.sigmoid(out)
        return out, hidden
    
