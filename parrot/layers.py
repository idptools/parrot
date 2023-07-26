import torch
import torch.nn as nn

class ToMatrixLayer(nn.Module):
    def __init__(self, input_size, seq_len):
        super(ToMatrixLayer, self).__init__()
         # Linear layer to predict triangular matrix
        self.linear = nn.Linear(input_size, input_size * (input_size + 1) // 2) 

    def forward(self, x):
        batch_size, input_size = x.size(0), x.size(1)
        # Predict triangular matrix
        out = self.linear(x)  

        # Convert triangular matrix to full symmetric matrix
        symmetric_matrix = torch.zeros(batch_size, input_size, input_size)
        
        # Upper triangular portion
        symmetric_matrix[:, torch.triu_indices(input_size)] = out  
        # Reflect and fill lower triangular portion
        symmetric_matrix = symmetric_matrix + symmetric_matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(symmetric_matrix, dim1=-2, dim2=-1)) 
        
        return symmetric_matrix
