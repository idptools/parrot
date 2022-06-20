"""
The underlying architecture of the transformer network used in PARROT

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""
import torch
import torch.nn as nn

class Transformer_MtM(nn.Module):
    """A Pytorch many-to-many Transformer neural network

    """

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, activation, device, batch_first=True):
        """
        Parameters
        ----------
        d_model : int
            The number of expected features in the encoder/decoder inputs

        nhead : int
            Size of hidden vectors in the network

        num_encoder_layers : int
            The number of sub-encoder-layers in the encoder.

        num_decoder_layers : int
            The number of sub-decoder-layers in the decoder.
    
        dim_feedforward : int
            The dimension of the feedforward network model

        dropout : int
            The dropout value

        activation : str
            The activation function of encoder/decoder intermediate layer,
            can be a string (“relu” or “gelu”) or a unary callable. 

        device : str
            String describing where the network is physically stored on the computer.
            Should be either 'cpu' or 'cuda' (GPU).

        batch_first : boolean
            If True, (batch, seq, feature).
            If False, (seq, batch, feature).
            Default = True
        """
        super(Transformer_MtM, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.device = device
        self.batch_first = batch_first
        self.transformer = nn.Transformer(
                                        d_model=d_model,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        activation=activation,
                                        batch_first=batch_first,
                                        )

    def forward(self, x):
        """Propogate input sequences through the network to produce outputs

        Parameters
        ----------
        x : 3-dimensional PyTorch IntTensor
            
        Returns
        -------
        3-dimensional PyTorch FloatTensor
           
        *** UPDATE THIS *** is this true? Need trial inputs for this.
        """


