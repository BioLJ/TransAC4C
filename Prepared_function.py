import math
from transformer import *
class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=10000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb
###### Positional
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 15000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)


    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)
############# Encoder

#############
class CNNnet2(nn.Module):
    def __init__(self, in_channels, out_channels,padding):
        super(CNNnet, self).__init__()
        self.cnn1= nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2,2),
            nn.Dropout(0.5))
        self.cnn2=nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, stride=1, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2,2),
            nn.Dropout(0.5))

        self.Lstmmodel1=nn.LSTM(input_size=8,hidden_size=16,batch_first=True,bidirectional=True)
        self.Lstmmodel2=nn.LSTM(input_size=8,hidden_size=16,batch_first=True,bidirectional=True)

    def forward(self, x):
        out1=self.cnn1(x)
        out1=out1.permute(0,2,1)
        print(out1.size())
        out1, (h_n, c_n)= self.Lstmmodel(out1)
        out1=out1.reshape(out1.size()[0], -1)
        print(out1.size())
        return out1
#############
class CNNnet(nn.Module):
    def __init__(self, in_channels, out_channels,padding):
        super(CNNnet, self).__init__()
        self.cnn1= nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2,2),
            nn.Dropout(0.5))
        
        self.Lstmmodel=nn.LSTM(input_size=300,hidden_size=64,batch_first=True,bidirectional=True)
        
    def forward(self, x):
        out, (h_n, c_n)= self.Lstmmodel(x)
        out=out.permute(0,2,1)
        out=self.cnn1(out)
        out=out.permute(0,2,1)
        #print(out.size())
        out=out.reshape(out.size()[0], -1)
        #print(out.size())
        return out
######
class AC4Cer(nn.Module):
    def __init__(self, d_model=300, n_spk=2):
        super().__init__()
        self.embedding=nn.Sequential(nn.Embedding(413,d_model),
                                     nn.Dropout(0))
        self.pos=LearnedPositionEncoding(d_model=d_model)
        self.pos2=PositionalEncoding(dropout=0,dim=300,max_len=512)
        self.encod=TransformerEncoderLayer(d_model=d_model,
                                                dim_feedforward=2048,
                                                nhead=6)
        self.encoder=TransformerEncoder(self.encod,num_layers=1)
        self.cnn=CNNnet(in_channels=d_model,out_channels=256,padding=0)
        self.FC=nn.Sequential(
            nn.Linear(1600,256*4),
            nn.ELU(),
            nn.Linear(256*4,64),
            nn.ELU(),
            nn.Linear(64,2)

        )
        self.FCtry=nn.Sequential(
            nn.Linear(413,30),
            nn.ReLU(),
            nn.Linear(30,2)
        )
    def forward(self,x):
        out=self.embedding(x).permute(1,0,2)
        #out=self.pos2(out).permute(1,0,2)
        #out=self.pos(out).permute(1,0,2)
        out,w=self.encoder(out)
        out=out.transpose(0,1)
        #out=out.mean(dim=2)
        #out=out.unsqueeze(1)
        #out=out.permute(0,2,1)
        #print(out.size())
        out=self.cnn(out)
        #print(out.size())
        #print(out.size())
        out=self.FC(out)
        return out,w
######### model_long
class CNNnet_long(nn.Module):
    def __init__(self, in_channels, out_channels,padding):
        super(CNNnet_long, self).__init__()
        self.cnn1= nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),  
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2,2),
            nn.Dropout(0.1))
        
        self.Lstmmodel=nn.LSTM(input_size=300,hidden_size=64,batch_first=True,bidirectional=True)
        
    def forward(self, x):
        out, (h_n, c_n)= self.Lstmmodel(x)
        out=out.permute(0,2,1)
        out=self.cnn1(out)
        out=out.permute(0,2,1)
        #print(out.size())
        out=out.reshape(out.size()[0], -1)
        #print(out.size())
        return out
######
class AC4Cer_Long(nn.Module):
    def __init__(self, d_model=300, n_spk=2):
        super().__init__()
        self.embedding=nn.Sequential(nn.Embedding(413,d_model),
                                     nn.Dropout(0))
        self.pos=LearnedPositionEncoding(d_model=d_model)
        self.pos2=PositionalEncoding(dropout=0,dim=300,max_len=512)
        self.encod=TransformerEncoderLayer(d_model=d_model,
                                                dim_feedforward=2048,
                                                nhead=6)
        self.encoder=TransformerEncoder(self.encod,num_layers=1)
        self.cnn=CNNnet_long(in_channels=d_model,out_channels=256,padding=0)
        self.FC=nn.Sequential(
            nn.Linear(24,2)
        )

    def forward(self,x):
        out=self.embedding(x).permute(1,0,2)

        out,w=self.encoder(out)
        out=out.transpose(0,1)
        out=self.cnn(out)
        out=self.FC(out)
        return out,w
