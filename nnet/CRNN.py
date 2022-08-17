import torch
from torch import nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            torch.nn.MaxPool2d((1, 5), stride=(1, 5))
        )
            
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            torch.nn.MaxPool2d((1, 2), stride=(1, 2))     
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            torch.nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def forward(self, x):

        
        o_1 = self.cnn1(x)
        o_2 = self.cnn2(o_1)
        o_3 = self.cnn3(o_2)
        o_4 =o_3.permute(0,2,1,3)
        o_5=torch.cat([o_4[:,:,:,0],o_4[:,:,:,1]],dim=2) #(batch_size, n_frames, n_channels & n_freq)
        return o_5

class BidirectionalGRU(nn.Module):
    def __init__(self, n_in = 256, n_hidden = 16, dropout=0, num_layers=2):

        """
            Initialization of BidirectionalGRU instance
        Args:
            n_in: int, number of input
            n_hidden: int, number of hidden layers
            dropout: flat, dropout
            num_layers: int, number of layers
        """

        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(
            n_in,
            n_hidden,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
            num_layers=num_layers,
        )

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent


class CRNN(nn.Module):
  def __init__(self):
    super(CRNN, self).__init__()
    self.cnn = CNN()
    self.rnn = BidirectionalGRU()
    self.dropout = nn.Dropout(0.5)
    self.dense = nn.Linear(32, 3)

    self.sigmoid = nn.Sigmoid()
    

  def forward(self, x):
    x = x.transpose(1, 2).unsqueeze(1)
    
    #input size : (batch_size, n_channels, n_frames/n_segment, n_freq) <- mel spectogram
    x = self.cnn(x)
    x = self.rnn(x)
    x = self.dropout(x)
    x = self.dense(x)
    strong = self.sigmoid(x)
    return strong