#!/usr/bin/env python

import torch 
import torch.nn as nn

# TODO: Add more extensive comments
# For regression, num_classes=1. For classification, num_classes=#classes.

# Device configuration
DEVICE = torch.device('cpu')
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Many-to-Many bidirectional recurrent neural network
class BRNN_MtM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN_MtM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
        					batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2, 	# *2 for bidirection
        					out_features=num_classes) 
    
    def forward(self, x):
        # dimensions of x: [batch_size X seq_len X input_vector]

        # Set initial states
        # h0 and c0 dimensions: [num_layers*2 X batch_size X hidden_size]
        h0 = torch.zeros(self.num_layers*2, 	# *2 for bidirection
        				 x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers*2, 
        				 x.size(0), self.hidden_size).to(DEVICE)
        
        # Forward propagate LSTM
        # out: tensor of shape: [batch_size, seq_length, hidden_size*2]
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Decode the hidden state for each time step
        fc_out = self.fc(out)      
        #fc_out = fc_out.view(x.size(0), -1, self.num_classes)
        return fc_out

# Many-to-One bidirectional recurrent neural network
class BRNN_MtO(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN_MtO, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
        					batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2, 	# *2 for bidirection
        					out_features=num_classes) 
    
    def forward(self, x):
        # dimensions of x: [batch_size X seq_len X input_vector]

        # Set initial states
        # h0 and c0 dimensions: [num_layers*2 X batch_size X hidden_size]
        h0 = torch.zeros(self.num_layers*2, 	# *2 for bidirection
        				 x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers*2, 
        				 x.size(0), self.hidden_size).to(DEVICE)
        
        # Forward propagate LSTM
        # out: tensor of shape: [batch_size, seq_length, hidden_size*2]
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Retain the outputs of the last time step in the sequence for both directions
        # (i.e. output of seq[n] in forward direction, seq[0] in reverse direction)
        final_outs = torch.cat((h_n[:, :, :][-2, :], h_n[:, :, :][-1, :]), -1)
		        
        # Decode the hidden state of the last time step
        fc_out = self.fc(final_outs)
        return fc_out
