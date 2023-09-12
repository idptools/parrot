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
from torchmetrics import Accuracy, MatthewsCorrCoef, R2Score, AUROC, F1Score, Precision
import numpy as np

import os
import datetime
import IPython

from parrot import process_input_data as pid
from parrot.tools import validate_args
from parrot.tools import dataset_warnings
from parrot.layers import ToMatrixLayer

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
        """A Pytorch Lightning DataModule for PARROT formatted data files. 
        This can be passed to a Pytorch Lightning Trainer object to train a PARROT network.

        Parameters
        ----------
        tsv_file : str
            The path to the training/validation/test split tsv file.
        num_classes : int
            Number of classes for the machine learning task, use 1 for regression.
        datatype : str
            Must be either 'residues' or 'sequence'
        batch_size : int, optional
            batch size to train your model with., by default 32
        encode : str, optional
            encoding scheme to convert from amino acids to numerical vector, by default 'onehot'
        fractions : list, optional
            train, validation, test splits, by default [0.6, 0.25, 0.15]
        split_file : _type_, optional
            The path to the file indicating the train/validation/test splits, by default None
        excludeSeqID : bool, optional
            Option to exclude the ID column, by default False
        ignore_warnings : bool, optional
            Ignore PARROT prompted warnings, by default False
        save_splits : bool, optional
            Optionally save the train/val/test splits, by default True
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
    
        # if true and split file has not been provided
        if self.save_splits and not os.path.isfile(self.split_file):
            # take TSV file
            network_file = os.path.abspath(self.tsv_file)
            # Extract tsv filename without the extension and parent directory of TSV
            filename_prefix, parent_dir = validate_args.split_file_and_directory(network_file)
            self.split_file = f"{filename_prefix}_split_file.txt"

        self.num_workers = os.cpu_count() if os.cpu_count() <= 32 else os.cpu_count()//8


    def prepare_data(self):
        pid.split_data(self.tsv_file, datatype=self.datatype, 
                                problem_type=self.problem_type,
                                num_classes=self.num_classes,
                                excludeSeqID=self.excludeSeqID, 
                                split_file=self.split_file, 
                                encoding_scheme=self.encoding_scheme, 
                                encoder=self.encoder, 
                                percent_val=self.fractions[1], 
                                percent_test=self.fractions[2],
                                ignoreWarnings=self.ignore_warnings,
                )        

    def setup(self, stage=None):
    
        self.train, self.val, self.test = pid.split_data(self.tsv_file, datatype=self.datatype, 
                                                          problem_type=self.problem_type,
                                                          num_classes=self.num_classes,
                                                          excludeSeqID=self.excludeSeqID, 
                                                          split_file=self.split_file, 
                                                          encoding_scheme=self.encoding_scheme, 
                                                          encoder=self.encoder, 
                                                          percent_val=self.fractions[1], 
                                                          percent_test=self.fractions[2],
                                                          ignoreWarnings=self.ignore_warnings,
                                                        )
    
    def train_dataloader(self):
        # Create and return the training dataloader
        return DataLoader(self.train, batch_size=self.batch_size, 
                          collate_fn=self.collate_function,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Create and return the validation dataloader
        return DataLoader(self.val, batch_size=self.batch_size, 
                            collate_fn=self.collate_function,
                            shuffle=False,
                            num_workers=self.num_workers)
    
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
    lstm_hidden_size : int
        Size of hidden vectors in the network
    num_lstm_layers : int
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

    def __init__(self, input_size, lstm_hidden_size, num_lstm_layers, 
                        num_classes, problem_type,
                        datatype, **kwargs):
        """
        Parameters
        ----------
        input_size : int
            Length of the input vectors at each timestep
        lstm_hidden_size : int
            Size of hidden vectors in the network
        num_lstm_layers : int
            Number of hidden layers (for each direction) in the network
        num_classes : int
            Number of classes for the machine learning task. If it is a regression
            problem, `num_classes` should be 1. If it is a classification problem,
            it should be the number of classes.
        """
        super(BRNN_MtM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.datatype = datatype
        self.problem_type = problem_type
        
        self.num_linear_layers = kwargs.get("num_linear_layers", 1)
        self.optimizer_name = kwargs.get('optimizer_name', 'SGD')
        self.linear_hidden_size = kwargs.get('linear_hidden_size', None)
        self.learn_rate = kwargs.get('learn_rate', 1e-3)
        self.dropout = kwargs.get('dropout', None)

        # Core Model architecture!
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_lstm_layers,
                                batch_first=True, bidirectional=True)
        
        # improve generalization, stability, and model capacity
        self.layer_norm = nn.LayerNorm(lstm_hidden_size*2)

        self.linear_layers = nn.ModuleList()
        # increase LSTM embedding to linear hidden size dimension * 2 because bidirection-LSTM
        for i in range(0,self.num_linear_layers):
            if i == 0 and i == self.num_linear_layers - 1:
                # if theres only one linear layer map to output (old parrot-style)
                self.linear_layers.append(nn.Linear(self.lstm_hidden_size*2, num_classes)) # *2 for bidirection LSTM
            elif i == 0:
                # if we're not going directly to output, add first layer to map to linear hidden size
                self.linear_layers.append(nn.Linear(self.lstm_hidden_size*2, self.linear_hidden_size)) 

                # add dropout on this initial layer if specified
                if self.dropout != 0.0 and self.dropout is not None:
                    self.linear_layers.append(nn.Dropout(self.dropout))
            elif i < self.num_linear_layers - 1:
                # if linear layer is even, add some dropout
                if i % 2 == 0 and self.dropout != 0.0:
                    self.linear_layers.append(nn.Linear(self.linear_hidden_size, self.linear_hidden_size))
                    self.linear_layers.append(nn.Dropout(self.dropout))
                    self.linear_layers.append(nn.ReLU())
                else:
                    # add second linear layer (index 1) to n-1. 
                    self.linear_layers.append(nn.Linear(self.linear_hidden_size, self.linear_hidden_size))
                    self.linear_layers.append(nn.ReLU())
            elif i == self.num_linear_layers - 1:
                # add final output layer
                self.linear_layers.append(nn.Linear(self.linear_hidden_size, num_classes))
            else:
                raise ValueError("Invalid number of linear layers. Must be greater than 0.")

        # set optimizer parameters
        if self.optimizer_name == "SGD":
            self.momentum = kwargs.get('momentum', 0.99)
        elif self.optimizer_name == "AdamW":
            self.beta1 = kwargs.get('beta1', 0.9)
            self.beta2 = kwargs.get('beta2', 0.999)
            self.eps = kwargs.get('eps', 1e-8)
            self.weight_decay = kwargs.get('weight_decay',1e-2)
        elif self.optimizer_name == "Adam":
            self.beta1 = kwargs.get('beta1', 0.9)
            self.beta2 = kwargs.get('beta2', 0.999)
            self.eps = kwargs.get('eps', 1e-8)
            self.weight_decay = kwargs.get('weight_decay',1e-2)

        # Set loss criteria
        if self.problem_type == 'regression':
            self.r2_score = R2Score(compute_on_cpu=True)
            if self.datatype == 'residues':
                # self.criterion = nn.MSELoss(reduction='mean')
                self.criterion = nn.MSELoss(reduction='sum')
            elif self.datatype == 'sequence':
                self.criterion = nn.L1Loss(reduction='sum')
        elif self.problem_type == 'classification':
            if self.num_classes > 2:
                self.task = 'multiclass'
                self.criterion = nn.CrossEntropyLoss(reduction='sum')
            else:
                self.task = 'binary'
                self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
            
            self.accuracy = Accuracy(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.precision = Precision(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.auroc = AUROC(task=self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.mcc = MatthewsCorrCoef(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.f1_score = F1Score(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
        else:
            raise ValueError("Invalid problem type. Supported options: 'regression', 'classification'.")

        # these are used to monitor the training losses for the *EPOCH*
        self.train_step_losses = []

        # save them sweet sweet hyperparameters 
        self.save_hyperparameters()

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
        # out: tensor of shape: [batch_size, seq_length, lstm_hidden_size*2]
        out, (h_n, c_n) = self.lstm(x)
        out = self.layer_norm(out)
        for layer in self.linear_layers:
            out = layer(out)

        return out

    def training_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
        
        self.train_step_losses.append(loss)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.train_step_losses).mean()
        self.log("epoch_train_loss", epoch_mean, prog_bar=True)
        
        # free up the memory
        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
            self.r2_score(outputs.view(-1,1), targets.float().view(-1,1))
            self.log('epoch_val_rsquare', self.r2_score)
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())   

            accuracy = self.accuracy(outputs, targets.long())
            self.log('epoch_val_accuracy', accuracy, on_step=True)
            
            f1score = self.f1_score(outputs, targets.long())
            self.log('epoch_val_f1score', f1score, on_step=True)

            auroc = self.auroc(outputs, targets.long())
            self.log('epoch_val_auroc', auroc, on_step=True)

            precision = self.precision(outputs, targets.long())
            self.log('epoch_val_precision', precision, on_step=True)

            mcc = self.mcc(outputs, targets.long())
            
            self.log('epoch_val_mcc', mcc, on_step=True)

        self.log('epoch_val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.learn_rate, momentum=self.momentum, nesterov=True)            
        # at some point fused=True in AdamW will be better but it LOOKS a little buggy right now - July 2023
        elif self.optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.learn_rate, betas=(self.beta1, self.beta2), 
                                                        eps=self.eps, weight_decay=self.weight_decay)
        elif self.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learn_rate, betas=(self.beta1, self.beta2), 
                                                        eps=self.eps, weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer name. Supported options: 'SGD', 'AdamW'.")
        
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
    lstm_hidden_size : int
        Size of hidden vectors in the network
    num_lstm_layers : int
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

    def __init__(self, input_size, lstm_hidden_size,
                  num_lstm_layers, num_classes, problem_type, datatype, **kwargs):
        """
        Parameters
        ----------
        input_size : int
            Length of the input vectors at each timestep
        lstm_hidden_size : int
            Size of hidden vectors in the network
        num_lstm_layers : int
            Number of hidden layers (for each direction) in the network
        num_classes : int
            Number of classes for the machine learning task. If it is a regression
            problem, `num_classes` should be 1. If it is a classification problem,
            it should be the number of classes.
        """

        super(BRNN_MtO, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.datatype = datatype
        self.problem_type = problem_type
        
        self.num_linear_layers = kwargs.get("num_linear_layers", 1)
        self.optimizer_name = kwargs.get('optimizer_name', 'SGD')
        self.linear_hidden_size = kwargs.get('linear_hidden_size', None)
        self.learn_rate = kwargs.get('learn_rate', 1e-3)
        self.dropout = kwargs.get('dropout', None)

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_lstm_layers,
                                batch_first=True, bidirectional=True)
        
        # improve generalization, stability, and model capacity
        # self.layer_norm = nn.LayerNorm(lstm_hidden_size*2)

        self.linear_layers = nn.ModuleList()
        # increase LSTM embedding to linear hidden size dimension * 2 because bidirection-LSTM
        for i in range(0,self.num_linear_layers):
            if i == 0 and i == self.num_linear_layers - 1:
                # if theres only one linear layer map to output (old parrot-style)
                self.linear_layers.append(nn.Linear(self.lstm_hidden_size*2, num_classes)) # *2 for bidirection LSTM
            elif i == 0:
                # if we're not going directly to output, add first layer to map to linear hidden size
                self.linear_layers.append(nn.Linear(self.lstm_hidden_size*2, self.linear_hidden_size)) 

                # add dropout on this initial layer if specified
                if self.dropout != 0.0 and self.dropout is not None:
                    self.linear_layers.append(nn.Dropout(self.dropout))
            elif i < self.num_linear_layers - 1:
                # if linear layer is even, add some dropout
                if i % 2 == 0 and self.dropout != 0.0:
                    self.linear_layers.append(nn.Linear(self.linear_hidden_size, self.linear_hidden_size))
                    self.linear_layers.append(nn.Dropout(self.dropout))
                    self.linear_layers.append(nn.ReLU())
                else:
                    # add second linear layer (index 1) to n-1. 
                    self.linear_layers.append(nn.Linear(self.linear_hidden_size, self.linear_hidden_size))
                    self.linear_layers.append(nn.ReLU())
            elif i == self.num_linear_layers - 1:
                # add final output layer
                self.linear_layers.append(nn.Linear(self.linear_hidden_size, num_classes))
            else:
                raise ValueError("Invalid number of linear layers. Must be greater than 0.")

        # set optimizer parameters
        if self.optimizer_name == "SGD":
            self.momentum = kwargs.get('momentum', 0.99)
        elif self.optimizer_name == "AdamW":
            self.beta1 = kwargs.get('beta1', 0.9)
            self.beta2 = kwargs.get('beta2', 0.999)
            self.eps = kwargs.get('eps', 1e-8)
            self.weight_decay = kwargs.get('weight_decay',1e-2)

        # Set loss criteria
        if self.problem_type == 'regression':
            self.r2_score = R2Score(compute_on_cpu=True)
            if self.datatype == 'residues':
                # self.criterion = nn.MSELoss(reduction='mean')
                self.criterion = nn.MSELoss(reduction='sum')
            elif self.datatype == 'sequence':
                self.criterion = nn.L1Loss(reduction='sum')
        elif self.problem_type == 'classification':
            if self.num_classes > 2:
                self.task = 'multiclass'
                self.criterion = nn.CrossEntropyLoss(reduction='sum')
            else:
                self.task = 'binary'
                self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
            
            self.accuracy = Accuracy(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.precision = Precision(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.auroc = AUROC(task=self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.mcc = MatthewsCorrCoef(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.f1_score = F1Score(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
        else:
            raise ValueError("Invalid problem type. Supported options: 'regression', 'classification'.")

        # set optimizer parameters
        if self.optimizer_name == "SGD":
            self.momentum = kwargs.get('momentum', 0.99)
        elif self.optimizer_name == "AdamW":
            self.beta1 = kwargs.get('beta1', 0.9)
            self.beta2 = kwargs.get('beta2', 0.999)
            self.eps = kwargs.get('eps', 1e-8)
            self.weight_decay = kwargs.get('weight_decay',1e-2)
        elif self.optimizer_name == "Adam":
            self.beta1 = kwargs.get('beta1', 0.9)
            self.beta2 = kwargs.get('beta2', 0.999)
            self.eps = kwargs.get('eps', 1e-8)
            self.weight_decay = kwargs.get('weight_decay',1e-2)

        # Set loss criteria
        if self.problem_type == 'regression':
            self.r2_score = R2Score(compute_on_cpu=True)
            if self.datatype == 'residues':
                # self.criterion = nn.MSELoss(reduction='mean')
                self.criterion = nn.MSELoss(reduction='sum')
            elif self.datatype == 'sequence':
                self.criterion = nn.L1Loss(reduction='sum')
        
        elif self.problem_type == 'classification':
            if self.num_classes > 2:
                self.task = 'multiclass'
                self.criterion = nn.CrossEntropyLoss(reduction='sum')
            else:
                self.task = 'binary'
                self.criterion = nn.BCEWithLogitLoss(reduction='sum')
            
            self.accuracy = Accuracy(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.auroc = AUROC(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.f1_score = F1Score(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.precision = Precision(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            self.mcc = MatthewsCorrCoef(task = self.task, num_classes=self.num_classes, compute_on_cpu=True)
            
        else:
            raise ValueError("Invalid problem type. Supported options: 'regression', 'classification'.")

        # these are used to monitor the training losses for the *EPOCH*
        self.train_step_losses = []
        self.val_step_losses = []

        # save them sweet sweet hyperparameters 
        self.save_hyperparameters()
    
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
        # out: tensor of shape: [batch_size, seq_length, lstm_hidden_size*2]
        out, (h_n, c_n) = self.lstm(x)
    
        # Retain the outputs of the last time step in the sequence for both directions
        # (i.e. output of seq[n] in forward direction, seq[0] in reverse direction)
        # forward_last_step = h_n[-2, :, :]
        # reverse_last_step = h_n[-1, :, :]
        out = torch.cat((h_n[:, :, :][-2, :], h_n[:, :, :][-1, :]), -1)

        # out = self.layer_norm(out)
        for layer in self.linear_layers:
            out = layer(out)

        return out

    def training_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
        self.train_step_losses.append(loss)

        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        my_data = np.array([data.detach().cpu().numpy() for data in self.train_step_losses])
        
        # train_step_losses 
        print("mydata.shape",my_data.shape)
        print("flattened_shape", my_data.flatten().shape)
        per_data_point_loss = my_data.sum()  / len(my_data.flatten())
        print("per_data_point step",per_data_point_loss.shape)
        print("per_data point val",per_data_point_loss) 

        epoch_mean = torch.stack(self.train_step_losses).mean()
        self.log("epoch_train_loss", epoch_mean, prog_bar=True)
        print("end of epoch mean: ", epoch_mean)
        # free up the memory
        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
            self.r2_score(outputs.view(-1,1), targets.float().view(-1,1))
            self.log('epoch_val_rsquare', self.r2_score)
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())   

            accuracy = self.accuracy(outputs, targets.long())
            self.log('epoch_val_accuracy', accuracy)
            
            f1score = self.f1_score(outputs, targets.long())
            self.log('epoch_val_f1score', f1score)

            auroc = self.auroc(outputs, targets.long())
            self.log('epoch_val_auroc', auroc)

            precision = self.precision(outputs, targets.long())
            self.log('epoch_val_precision', precision)
            
            mcc = self.mcc(outputs, targets.long())
            self.log('epoch_val_mcc', mcc)
        
        self.log('step_val_loss', loss, on_step=True)
        self.log('epoch_val_loss', loss / 4)

        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.learn_rate, momentum=self.momentum, nesterov=True)            
        # fused=True argument in AdamW will be much faster, but it LOOKS a little buggy right now - July 2023
        elif self.optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.learn_rate, betas=(self.beta1, self.beta2), 
                                                        eps=self.eps, weight_decay=self.weight_decay)
        elif self.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learn_rate, betas=(self.beta1, self.beta2), 
                                                        eps=self.eps, weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer name. Supported options: 'SGD', 'AdamW'.")
        
        return optimizer


#TODO continue fleshing out matrix functionality
class BRNN_Matrix(L.LightningModule):
    """A PyTorch model to predict a matrix of values from sequence

    Attributes
    ----------
    lstm_hidden_size : int
        Size of hidden vectors in the network
    num_lstm_layers : int
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

    def __init__(self, input_size, lstm_hidden_size, num_lstm_layers, 
                        num_classes, problem_type,
                        datatype, optimizer_name="SGD", **kwargs):
        """
        Parameters
        ----------
        input_size : int
            Length of the input vectors at each timestep
        lstm_hidden_size : int
            Size of hidden vectors in the network
        num_lstm_layers : int
            Number of hidden layers (for each direction) in the network
        num_classes : int
            Number of classes for the machine learning task. If it is a regression
            problem, `num_classes` should be 1. If it is a classification problem,
            it should be the number of classes.
        """

        super(BRNN_MtM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        
        self.datatype = datatype
        self.problem_type = problem_type
        self.optimizer_name = optimizer_name
        self.learn_rate = kwargs.get('learn_rate', 1e-3)

        # set optimizer parameters
        if self.optimizer_name == "SGD":
            self.momentum = kwargs.get('momentum', 0.99)
        elif self.optimizer_name == "Adam":
            self.beta1 = kwargs.get('beta1', 0.9)
            self.beta2 = kwargs.get('beta2', 0.999)
            self.eps = kwargs.get('eps', 1e-8)
            self.weight_decay = kwargs.get('weight_decay', 0)
        elif self.optimizer_name == "AdamW":
            self.beta1 = kwargs.get('beta1', 0.9)
            self.beta2 = kwargs.get('beta2', 0.999)
            self.eps = kwargs.get('eps', 1e-8)
            self.weight_decay = kwargs.get('weight_decay',1e-2)

        # Set loss criteria
        if self.problem_type == 'regression':
            self.r2_score = R2Score(compute_on_cpu=True)
            if self.datatype == 'residues' or self.problem_type == "matrix":
                self.criterion = nn.MSELoss(reduction='sum')
            elif self.datatype == 'sequence':
                self.criterion = nn.L1Loss(reduction='sum')
        elif self.problem_type == 'classification':
            self.accuracy = Accuracy(compute_on_cpu=True)
            self.criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            raise ValueError("Invalid problem type. Supported options: 'regression', 'classification'.")

        # these are used to monitor the training and validation losses for the *EPOCH*
        self.train_step_losses = []
        self.val_step_losses = []

        # Core Model architecture!
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_lstm_layers,
                                batch_first=True, bidirectional=True)
        
        # improve generalization, stability, and model capacity
        self.layer_norm = nn.LayerNorm(lstm_hidden_size*2)

        if self.num_linear_layers == 1:
            self.output_layer = nn.Linear(in_features=lstm_hidden_size*2,  # *2 for bidirection
                                        out_features=num_classes)
        elif self.num_linear_layers == 2:
            self.linear_layers = nn.ModuleList()
            self.linear_layers.append(nn.Linear(lstm_hidden_size*2, self.linear_hidden_size)) 
            self.output_layer = nn.Linear(self.linear_hidden_size, num_classes)
        else:
            self.linear_layers = nn.ModuleList()
            # increase LSTM embedding to linear hidden size dimension * 2 because bidirection-LSTM
            # add first layer
            self.linear_layers.append(nn.Linear(lstm_hidden_size*2, self.linear_hidden_size)) 
            
            # add layers 2 to n-1, this is only used if num_linear_layers > 2 because first and last layer is predefined
            for i in range(1, self.num_linear_layers-1):
                self.linear_layers.append(nn.ReLU(nn.Linear(self.linear_hidden_size, self.linear_hidden_size)))
                if i % 2 == 1 and self.dropout:
                    self.linear_layers.append(nn.Dropout(self.dropout))
            
            # add final output layer
            self.output_layer = nn.Linear(self.linear_hidden_size, num_classes)
        
        # save them sweet sweet hyperparameters 
        self.save_hyperparameters()

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
        # out: tensor of shape: [batch_size, seq_length, lstm_hidden_size*2]
        out, (h_n, c_n) = self.lstm(x)
        out = self.layer_norm(out)
        if self.num_linear_layers > 1:
            for layer in self.linear_layers:
                out = layer(out)
            out = self.output_layer(out)
        else:
            # Decode the hidden state for each time step
            out = self.output_layer(out)
    
        return out

    def training_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
        
        self.train_step_losses.append(loss)

        self.log('batch_train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(self.train_step_losses).mean()
        self.log("average_epoch_train_loss", epoch_mean, prog_bar=True)
        
        # free up the memory
        self.train_step_losses.clear()


    def validation_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors)
        if self.problem_type == 'regression':
            loss = self.criterion(outputs, targets.float())
            self.r2_score(outputs.view(-1,1), targets.float().view(-1,1))
            self.log('batch_val_rsquare', self.r2_score)
        else:
            if self.datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())   
            accuracy = self.accuracy(outputs, targets.long())
            self.log('batch_val_accuracy', accuracy)

        self.log('batch_val_loss', loss,)
        self.val_step_losses.append(loss)
        
        return loss
    
    def on_validation_epoch_end(self):
        # compute the average validation loss for the epoch
        mean_epoch_val_loss = torch.stack(self.val_step_losses).mean()
        self.log("average_epoch_val_loss", mean_epoch_val_loss, prog_bar=True)
        
        # free up the memory
        self.val_step_losses.clear()

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.learn_rate, momentum=self.momentum, nesterov=True)

        elif self.optimizer_name == "AdamW":
            # at some point this can be set to fused=True, but for now it looks a little buggy - July 2023
            optimizer = optim.AdamW(self.parameters(), lr=self.learn_rate,
                                     betas=(self.beta1, self.beta2), 
                                     eps=self.eps,
                                     weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer name. Supported options: 'SGD', 'Adam', 'AdamW'.")
        return optimizer


