"""
The underlying architecture of the bidirectional LSTM network used in PARROT

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""
import torch
import torch.nn as nn
import pytorch_lightning as L
# import lightning as L
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import os
import datetime
import IPython

from parrot import process_input_data as pid
from parrot import brnn_architecture
from parrot import train_network
from parrot import brnn_plot
from parrot.tools import validate_args
from parrot.tools import dataset_warnings

class ParrotDataModule(L.LightningDataModule):
    def __init__(self, tsv_file, 
                        num_classes, 
                        datatype, 
                        batch_size=32, 
                        encode='onehot', 
                        fractions=[0.6, 0.25, 0.15], 
                        split_file=None,
                        excludeSeqID=False,
                        ignore_warnings=False,
                        save_splits=True,
                 ):
        """_summary_

        Parameters
        ----------
        tsv_file : _type_
            _description_
        num_classes : _type_
            _description_
        datatype : str
            residues or sequence
        batch_size : int, optional
            _description_, by default 32
        encode : str, optional
            _description_, by default 'onehot'
        fractions : list, optional
            _description_, by default [0.6, 0.25, 0.15]
        split_file : _type_, optional
            _description_, by default None
        excludeSeqID : bool, optional
            _description_, by default False
        ignore_warnings : bool, optional
            _description_, by default False
        save_splits : bool, optional
            _description_, by default True
        """
        super().__init__()
        self.tsv_file = tsv_file
        self.num_classes = num_classes
        self.datatype = datatype
        self.batch_size = batch_size
        self.encode = encode

        self.problem_type, self.collate_function = validate_args.set_ml_task(self.num_classes, self.datatype)
        self.encoding_scheme, self.encoder, self.input_size = validate_args.set_encoding_scheme(self.encode)

        self.fractions = fractions
        
        self.split_file = split_file
        self.excludeSeqID = excludeSeqID
        self.ignore_warnings = ignore_warnings
        self.save_splits = save_splits


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Extract output directory and output prediction file name
        network_file = os.path.abspath(self.tsv_file)
        filename_prefix, output_dir = validate_args.split_file_and_directory(network_file)

        # If provided, check that split_file exists
        # if self.split_file is not None:
        current_date = datetime.date.today().strftime("%Y_%m_%d")
        self.save_splits_output = f"{filename_prefix}_{current_date}_split_file.txt"
        self.split_file = f"{filename_prefix}_{current_date}_split_file.txt"
    
        self.split_file = validate_args.check_file_exists(self.split_file, 'Split-file')
        # else:
            # self.split_file = None

        # If specified, get location where randomly generated train/val/test splits will be saved
        # if self.save_splits:
        
        # else:
            # self.save_splits_output = None

        self.train, self.val, self.test = pid.split_data(self.tsv_file, 
                                                    datatype=self.datatype, 
                                                    problem_type=self.problem_type,
                                                    num_classes=self.num_classes,
                                                    excludeSeqID=self.excludeSeqID, 
                                                    split_file=self.split_file, 
                                                    encoding_scheme=self.encoding_scheme, 
                                                    encoder=self.encoder, 
                                                    percent_val=self.fractions[1], 
                                                    percent_test=self.fractions[2],
                                                    ignoreWarnings=self.ignore_warnings,
                                                    save_splits_output=self.save_splits_output)
        # # redefine split_file
        

    def train_dataloader(self):
        # Create and return the training dataloader
        return DataLoader(self.train, batch_size=self.batch_size, 
                          collate_fn=self.collate_function,
                          shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        # Create and return the validation dataloader
        return DataLoader(self.val, batch_size=self.batch_size, 
                            collate_fn=self.collate_function,
                            shuffle=False,
                            num_workers=os.cpu_count())
    
    def test_dataloader(self):
        # Create and return the test dataloader
        return DataLoader(self.test, batch_size=1, collate_fn=self.collate_function)

class BRNN_MtM(L.LightningModule):
    """A PyTorch many-to-many bidirectional recurrent neural network

    A class containing the PyTorch implementation of a BRNN. The network consists
    of repeating LSTM units in the hidden layers that propogate sequence information
    in both the foward and reverse directions. A final fully connected layer
    aggregates the deepest hidden layers of both directions and produces the
    outputs.

    "Many-to-many" refers to the fact that the network will produce outputs 
    corresponding to every item of the input sequence. For example, an input 
    sequence of length 10 will produce 10 sequential outputs.

    Attributes
    ----------
    hidden_size : int
        Size of hidden vectors in the network
    num_layers : int
        Number of hidden layers (for each direction) in the network
    num_classes : int
        Number of classes for the machine learning task. If it is a regression
        problem, `num_classes` should be 1. If it is a classification problem,
        it should be the number of classes.
    lstm : PyTorch LSTM object
        The bidirectional LSTM layer(s) of the recurrent neural network.
    fc : PyTorch Linear object  
        The fully connected linear layer of the recurrent neural network. Across 
        the length of the input sequence, this layer aggregates the output of the
        LSTM nodes from the deepest forward layer and deepest reverse layer and
        returns the output for that residue in the sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers, 
                        num_classes, problem_type,
                        datatype, learn_rate):
        """
        Parameters
        ----------
        input_size : int
            Length of the input vectors at each timestep
        hidden_size : int
            Size of hidden vectors in the network
        num_layers : int
            Number of hidden layers (for each direction) in the network
        num_classes : int
            Number of classes for the machine learning task. If it is a regression
            problem, `num_classes` should be 1. If it is a classification problem,
            it should be the number of classes.
        """

        super(BRNN_MtM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.learn_rate = learn_rate
        self.datatype = datatype
        self.problem_type = problem_type

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2,  # *2 for bidirection
                            out_features=num_classes)
        self.save_hyperparameters()

        # Set loss criteria
        if self.problem_type == 'regression':
            if self.datatype == 'residues':
                self.criterion = nn.MSELoss(reduction='sum')
            elif self.datatype == 'sequence':
                self.criterion = nn.L1Loss(reduction='sum')
        elif self.problem_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x):
        """Propogate input sequences through the network to produce outputs

        Parameters
        ----------
        x : 3-dimensional PyTorch IntTensor
            Input sequence to the network. Should be in the format:
            [batch_dim X sequence_length X input_size]

        Returns
        -------
        3-dimensional PyTorch FloatTensor
            Output after propogating the sequences through the network. Will
            be in the format:
            [batch_dim X sequence_length X num_classes]
        """
        # Forward propagate LSTM
        # out: tensor of shape: [batch_size, seq_length, hidden_size*2]
        out, (h_n, c_n) = self.lstm(x)

        # Decode the hidden state for each time step
        fc_out = self.fc(out)
        return fc_out

    def training_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        return optimizer

class BRNN_MtO(L.LightningModule):
    """A PyTorch many-to-one bidirectional recurrent neural network

    A class containing the PyTorch implementation of a BRNN. The network consists
    of repeating LSTM units in the hidden layers that propogate sequence information
    in both the foward and reverse directions. A final fully connected layer
    aggregates the deepest hidden layers of both directions and produces the
    output.

    "Many-to-one" refers to the fact that the network will produce a single output 
    for an entire input sequence. For example, an input sequence of length 10 will
    produce only one output.

    Attributes
    ----------
    hidden_size : int
        Size of hidden vectors in the network
    num_layers : int
        Number of hidden layers (for each direction) in the network
    num_classes : int
        Number of classes for the machine learning task. If it is a regression
        problem, `num_classes` should be 1. If it is a classification problem,
        it should be the number of classes.
    lstm : PyTorch LSTM object
        The bidirectional LSTM layer(s) of the recurrent neural network.
    fc : PyTorch Linear object  
        The fully connected linear layer of the recurrent neural network. Across 
        the length of the input sequence, this layer aggregates the output of the
        LSTM nodes from the deepest forward layer and deepest reverse layer and
        returns the output for that residue in the sequence.
    """

    def __init__(self, input_size, hidden_size, 
                            num_layers, num_classes,
                            problem_type, datatype, learn_rate):
        """
        Parameters
        ----------
        input_size : int
            Length of the input vectors at each timestep
        hidden_size : int
            Size of hidden vectors in the network
        num_layers : int
            Number of hidden layers (for each direction) in the network
        num_classes : int
            Number of classes for the machine learning task. If it is a regression
            problem, `num_classes` should be 1. If it is a classification problem,
            it should be the number of classes.
        """

        super(BRNN_MtO, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learn_rate = learn_rate
        self.datatype = datatype
        self.problem_type = problem_type
        
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2,  # *2 for bidirection
                            out_features=num_classes)
        # Set loss criteria
        if self.problem_type == 'regression':
            if self.datatype == 'residues':
                self.criterion = nn.MSELoss(reduction='sum')
            elif self.datatype == 'sequence':
                self.criterion = nn.L1Loss(reduction='sum')
        elif self.problem_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, x):
        """Propogate input sequences through the network to produce outputs

        Parameters
        ----------
        x : 3-dimensional PyTorch IntTensor
            Input sequence to the network. Should be in the format:
            [batch_dim X sequence_length X input_size]

        Returns
        -------
        3-dimensional PyTorch FloatTensor
            Output after propogating the sequences through the network. Will
            be in the format:
            [batch_dim X 1 X num_classes]
        """
        # Forward propagate LSTM
        # out: tensor of shape: [batch_size, seq_length, hidden_size*2]
        out, (h_n, c_n) = self.lstm(x)

        # Retain the outputs of the last time step in the sequence for both directions
        # (i.e. output of seq[n] in forward direction, seq[0] in reverse direction)
        final_outs = torch.cat((h_n[:, :, :][-2, :], h_n[:, :, :][-1, :]), -1)

        # Decode the hidden state of the last time step
        fc_out = self.fc(final_outs)
        return fc_out


    def training_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        return optimizer