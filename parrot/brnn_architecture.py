"""
The underlying architecture of the bidirectional LSTM network used in PARROT

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import os

import pytorch_lightning as L
import torch
import torch.nn as nn

# import lightning as L
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MatthewsCorrCoef,
    MeanMetric,
    Precision,
    R2Score,
)

#from parrot import process_input_data as pid
from parrot import process_input_data as pid
from parrot.tools import validate_args


def _build_linear_layers(lstm_hidden_size, num_classes, num_linear_layers=1, 
                        linear_hidden_size=None, dropout=None):
    """Build linear layers for BRNN architectures to eliminate code duplication.
    
    Parameters
    ----------
    lstm_hidden_size : int
        Size of LSTM hidden layers (will be doubled for bidirectional)
    num_classes : int
        Number of output classes
    num_linear_layers : int, optional
        Number of linear layers, by default 1
    linear_hidden_size : int, optional
        Hidden size for intermediate linear layers, by default None
    dropout : float, optional
        Dropout rate, by default None
        
    Returns
    -------
    nn.ModuleList
        List of linear layers
    """
    linear_layers = nn.ModuleList()
    
    for i in range(num_linear_layers):
        if i == 0 and i == num_linear_layers - 1:
            # Single linear layer - map directly to output
            linear_layers.append(nn.Linear(lstm_hidden_size * 2, num_classes))
        elif i == 0:
            # First layer - map to hidden size
            if linear_hidden_size is None:
                raise ValueError("linear_hidden_size must be specified when num_linear_layers > 1")
            linear_layers.append(nn.Linear(lstm_hidden_size * 2, linear_hidden_size))
            linear_layers.append(nn.ReLU())
            # Add dropout
            if dropout is not None and dropout > 0.0:
                linear_layers.append(nn.Dropout(dropout))
            
        elif i < num_linear_layers - 1:
            # Intermediate layers
            linear_layers.append(nn.Linear(linear_hidden_size, linear_hidden_size))
            linear_layers.append(nn.ReLU())
            if dropout is not None and dropout > 0.0:
                linear_layers.append(nn.Dropout(dropout))
        elif i == num_linear_layers - 1:
            # Final output layer
            linear_layers.append(nn.Linear(linear_hidden_size, num_classes))
        else:
            raise ValueError("Invalid number of linear layers. Must be greater than 0.")
    
    return linear_layers


class ParrotDataModule(L.LightningDataModule):
    def __init__(
        self,
        tsv_file,
        num_classes,
        datatype,
        batch_size=32,
        encode="onehot",
        fractions=[0.6, 0.25, 0.15],
        split_file=None,
        excludeSeqID=False,
        ignore_warnings=False,
        save_splits=True,
        num_workers=None,
        distributed=False,
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
        distributed : bool, optional
            Set whether training is distributed. Default is False. 
        """
        super().__init__()
        
        # Input validation
        if not os.path.exists(tsv_file):
            raise FileNotFoundError(f"TSV file not found: {tsv_file}")
        
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        
        if datatype not in ['residues', 'sequence']:
            raise ValueError("datatype must be either 'residues' or 'sequence'")
        
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        
        if not isinstance(fractions, (list, tuple)) or len(fractions) != 3:
            raise ValueError("fractions must be a list/tuple of 3 values")
        
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")
        
        if any(f <= 0 for f in fractions):
            raise ValueError("All fractions must be positive")
        
        self.tsv_file = tsv_file
        self.num_classes = num_classes
        self.datatype = datatype
        self.batch_size = batch_size
        self.encode = encode
        self.distributed=distributed
        self.problem_type, self.collate_function = validate_args.set_ml_task(
            self.num_classes, self.datatype
        )
        self.encoding_scheme, self.encoder, self.input_size = (
            validate_args.set_encoding_scheme(self.encode)
        )
        self.fractions = fractions
        self.split_file = split_file
        self.excludeSeqID = excludeSeqID
        self.ignore_warnings = ignore_warnings
        self.save_splits = save_splits

        # set prepare_data_per_node depending on if distributed
        if self.distributed:
            self.prepare_data_per_node = False
        else:
            self.prepare_data_per_node = True

        # load dataset 
        self.dataset=pid.SequenceDataset(self.tsv_file)

        # if we don't have a name for split_file, make one. 
        if self.split_file==None:
            # take TSV file
            network_file = os.path.abspath(self.tsv_file)
            # Extract tsv filename without the extension and parent directory of TSV
            filename_prefix, parent_dir = validate_args.split_file_and_directory(
                network_file
            )
            self.split_file = f"{filename_prefix}_split_file.txt"

        if num_workers is not None:
            self.num_workers = num_workers
        else:
            self.num_workers = (
                os.cpu_count() if os.cpu_count() <= 32 else os.cpu_count() // 4
            )

    def prepare_data(self):
        pid.initial_data_prep(save_splits_loc = self.split_file, 
                                dataset=self.dataset, 
                                train_ratio=self.fractions[0], 
                                val_ratio=self.fractions[1])

    def setup(self, stage):
        self.train_indices, self.val_indices, self.test_indices = pid.read_indices(self.split_file)
        self.train_loader, self.val_loader, self.test_loader = pid.create_dataloaders(
                                                                    dataset=self.dataset,
                                                                    train_indices=self.train_indices,
                                                                    val_indices=self.val_indices,
                                                                    test_indices=self.test_indices,
                                                                    batch_size=self.batch_size)

    def train_dataloader(self):
        # Create and return the training dataloader
        return self.train_loader

    def val_dataloader(self):
        # Create and return the validation dataloader
        return self.val_loader

    def test_dataloader(self):
        # Create and return the test dataloader
        return self.test_loader



class BRNN_PARROT(L.LightningModule):
    """A PARROT BRNN that can be used for many-to-many or many-to-one problems.

    A class containing the PyTorch implementation of a BRNN. The network consists
    of repeating LSTM units in the hidden layers that propogate sequence information
    in both the foward and reverse directions. A final fully connected layer
    aggregates the deepest hidden layers of both directions and produces the
    outputs.

    "Many-to-many" refers to the fact that the network will produce outputs
    corresponding to every item of the input sequence. For example, an input
    sequence of length 10 will produce 10 sequential outputs.

    "Many-to-one" refers to the fact that the network will produce a single output
    for an entire input sequence. For example, an input sequence of length 10 will
    produce only one output.
    If `datatype` is 'sequence', a many-to-one problem will produce an output
    of shape [batch_size, num_classes]. If `datatype` is 'residues', a many-to-one
    problem will produce an output of shape [batch_size, sequence_length, num_classes].

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
    problem_type : str
        Type of problem to solve, either 'regression' or 'classification'.
    datatype : str
        Type of data being processed, either 'residues' or 'sequence'.
    batch_size : int
        Size of the batch for training and inference.
    **kwargs : dict
        Additional keyword arguments for model configuration, such as:
        - num_linear_layers: int, number of linear layers in the model
        - optimizer_name: str, name of the optimizer to use (default: 'SGD')
        - linear_hidden_size: int, size of hidden layers in the linear part of the model
        - learn_rate: float, learning rate for the optimizer (default: 1e-3)
        - dropout: float, dropout rate for the model (default: None)
        - direction: str, either 'minimize' or 'maximize', used for learning rate scheduler
        - monitor: str, metric to monitor for learning rate scheduler (default: 'epoch_val_loss')
        - distributed: bool, whether the model is being trained in a distributed setting (default: False)
        - momentum: float, momentum for SGD optimizer (default: 0.99)
        - beta1: float, first beta parameter for AdamW/Adam optimizer (default: 0.9)
        - beta2: float, second beta parameter for AdamW/Adam optimizer (default: 0.999)
        - eps: float, epsilon for AdamW/Adam optimizer (default: 1e-8)
        - weight_decay: float, weight decay for AdamW/Adam optimizer (default: 1e-2)
        
    """

    def __init__(
        self,
        input_size,
        lstm_hidden_size,
        num_lstm_layers,
        num_classes,
        problem_type,
        datatype,
        batch_size,
        **kwargs,
    ):
        super(BRNN_PARROT, self).__init__()
        
        # Input validation
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("input_size must be a positive integer")
        
        if not isinstance(lstm_hidden_size, int) or lstm_hidden_size < 1:
            raise ValueError("lstm_hidden_size must be a positive integer")
        
        if not isinstance(num_lstm_layers, int) or num_lstm_layers < 1:
            raise ValueError("num_lstm_layers must be a positive integer")
        
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")        

        if problem_type not in ['regression', 'classification']:
            raise ValueError("problem_type must be either 'regression' or 'classification'")
        
        if datatype not in ['residues', 'sequence']:
            raise ValueError("datatype must be either 'residues' or 'sequence'")
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.datatype = datatype
        self.problem_type = problem_type
        self.batch_size = batch_size

        self.num_linear_layers = kwargs.get("num_linear_layers", 1)
        self.optimizer_name = kwargs.get("optimizer_name", "SGD")
        self.linear_hidden_size = kwargs.get("linear_hidden_size", None)
        self.learn_rate = kwargs.get("learn_rate", 1e-3)
        self.dropout = kwargs.get("dropout", None)

        # set default monitor
        self.monitor = kwargs.get("monitor", "epoch_val_loss")

        # Core Model architecture!
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.dropout is not None and num_lstm_layers > 1 else 0.0
        )

        # improve generalization, stability, and model capacity
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)

        self.linear_layers = _build_linear_layers(lstm_hidden_size, num_classes, 
                                                self.num_linear_layers, 
                                                self.linear_hidden_size, 
                                                self.dropout)

        # set optimizer parameters
        if self.optimizer_name == "SGD":
            self.momentum = kwargs.get("momentum", 0.99)
        elif self.optimizer_name == "AdamW":
            self.beta1 = kwargs.get("beta1", 0.9)
            self.beta2 = kwargs.get("beta2", 0.999)
            self.eps = kwargs.get("eps", 1e-8)
            self.weight_decay = kwargs.get("weight_decay", 1e-2)
        else:
            raise ValueError(
                "Invalid optimizer name. Supported options: 'SGD', 'AdamW'."
            )
        
        # set if distributed
        self.distributed = kwargs.get("distributed",False)

        # set direction map...
        direction_map = {"minimize": "min", "maximize": "max"}
        # used for LR scheduler to min or max the LR upon plateau
        if kwargs.get("direction"):
            self.lr_direction = direction_map[kwargs.get("direction")]
        else:
            self.lr_direction = "min"

        self.monitor = kwargs.get("monitor", "epoch_val_loss")

        # Set loss criteria
        if self.problem_type == "regression":
            self.r2_score = R2Score(compute_on_cpu=True)
            if self.datatype == "residues":
                self.criterion = nn.MSELoss(reduction="mean")
            elif self.datatype == "sequence":
                self.criterion = nn.L1Loss(reduction="mean")
        elif self.problem_type == "classification":
            self.task = "multiclass"
            self.criterion = nn.CrossEntropyLoss(reduction="mean")

            self.accuracy = Accuracy(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.precision = Precision(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.auroc = AUROC(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.mcc = MatthewsCorrCoef(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.f1_score = F1Score(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
        else:
            raise ValueError(
                "Invalid problem type. Supported options: 'regression', 'classification'."
            )

        # these are used to monitor the training losses for the *EPOCH*
        self.train_loss_metric = MeanMetric()

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
            [batch_dim X sequence_length X num_classes] for MtM
            [batch_dim X num_classes] for MtO
        """
        # Forward propagate LSTM
        # out: tensor of shape: [batch_size, seq_length, lstm_hidden_size*2]
        out, (h_n, c_n) = self.lstm(x)
        
        # Different processing if MtM or MtO
        if self.datatype == "sequence":
            # Many-to-One: use only final hidden states from both directions
            # Retain the outputs of the last time step in the sequence for both directions
            # (i.e. output of seq[n] in forward direction, seq[0] in reverse direction)
            out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        
        # layer norm.
        out = self.layer_norm(out)
        
        # Apply linear layers
        for layer in self.linear_layers:
            out = layer(out)
        return out

    def training_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors.float())
        if self.problem_type == "regression":
            # Handle target reshaping for MtO regression
            if self.datatype=='sequence':
                targets = targets.view(-1, 1)
            loss = self.criterion(outputs, targets.float())
        else:
            if self.datatype == "residues":
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())

        self.train_loss_metric(loss)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        epoch_mean = self.train_loss_metric.compute()
        self.log("epoch_train_loss", epoch_mean, prog_bar=True,sync_dist=self.distributed)
        self.train_loss_metric.reset()

    def validation_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors.float())
        if self.problem_type == "regression":
            # Handle target reshaping for MtO regression
            if self.datatype == "sequence":
                targets = targets.view(-1, 1)
            loss = self.criterion(outputs, targets.float())
            # Only compute R² score if we have at least 2 samples
            if outputs.size(0) >= 2:
                self.r2_score(outputs.view(-1, 1), targets.float().view(-1, 1))
                self.log("epoch_val_rsquare", self.r2_score)
        else:
            if self.datatype == "residues":
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())

            accuracy = self.accuracy(outputs, targets.long())
            self.log("epoch_val_accuracy", accuracy, on_step=True)

            f1score = self.f1_score(outputs, targets.long())
            self.log("epoch_val_f1score", f1score, on_step=True)

            auroc = self.auroc(outputs, targets.long())
            self.log("epoch_val_auroc", auroc, on_step=True)

            precision = self.precision(outputs, targets.long())
            self.log("epoch_val_precision", precision, on_step=True)

            mcc = self.mcc(outputs, targets.long())
            self.log("epoch_val_mcc", mcc, on_step=True)

        self.log("epoch_val_loss", loss, prog_bar=True, sync_dist=self.distributed)
        self.log("val_loss", loss, sync_dist=self.distributed)  # For compatibility with standard monitoring
        return loss

    def test_step(self, batch, batch_idx):
        names, vectors, targets = batch
        outputs = self.forward(vectors.float())
        if self.problem_type == "regression":
            # if MtO
            if self.datatype== "sequence":
                # reshape targets for MtO regression
                targets = targets.view(-1, 1)
            # calc loss
            loss = self.criterion(outputs, targets.float())

            # Only compute R² score if we have at least 2 samples
            if outputs.size(0) >= 2:
                self.r2_score(outputs.view(-1, 1), targets.float().view(-1, 1))
                self.log("test_r2_score", self.r2_score)
        else:
            if self.datatype == "residues":
                outputs = outputs.permute(0, 2, 1)
            loss = self.criterion(outputs, targets.long())
            accuracy = self.accuracy(outputs, targets.long())
            self.log("test_accuracy", accuracy)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.learn_rate,
                momentum=self.momentum,
                nesterov=True,
            )

        # at some point fused=True in AdamW will be better but it LOOKS a little buggy right now - July 2023
        elif self.optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learn_rate,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                "Invalid optimizer name. Supported options: 'SGD', 'AdamW'."
            )

        lr_scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=0.0001
            ),
            "monitor": self.monitor,
            "interval": "epoch",
        }

        return [optimizer], [lr_scheduler]
    

class BRNN_PARROT_LEGACY(nn.Module):
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
    device : str
        String describing where the network is physically stored on the computer.
        Should be either 'cpu' or 'cuda' (GPU).
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

    def __init__(self, datatype, input_size, hidden_size, num_layers, num_classes, device):
        """
        Parameters
        ----------
        datatype: string
            if 'residues', the network will be used for residue-based data.
            If 'sequence', the network will be used for sequence-based data.
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
        device : str
            String describing where the network is physically stored on the computer.
            Should be either 'cpu' or 'cuda' (GPU).
        """

        super(BRNN_PARROT_LEGACY, self).__init__()
        self.datatype = datatype
        if datatype not in ['residues', 'sequence']:
            raise ValueError("datatype must be either 'residues' or 'sequence'")
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2,  # *2 for bidirection
                            out_features=num_classes)

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

        # Set initial states
        # h0 and c0 dimensions: [num_layers*2 X batch_size X hidden_size]
        h0 = torch.zeros(self.num_layers*2,     # *2 for bidirection
                         x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2,
                         x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # out: tensor of shape: [batch_size, seq_length, hidden_size*2]
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # If many-to-one, we only want the last output
        if self.datatype == "sequence":
            out=torch.cat((h_n[:, :, :][-2, :], h_n[:, :, :][-1, :]), -1)

        # Decode the hidden state for each time step
        fc_out = self.fc(out)
        return fc_out
    





"""
Code below for UNet that handles nXn input to nXn output predictions.

The UNet architecture takes nXn input and produces nXn output.
This makes it its own 'datatype' that is 'square' (not 'residues' or 'sequence').
It can handle both classification and regression problems.
"""

class DoubleConv(nn.Module):
    """Double convolution block used in UNet encoder and decoder paths.

    This block consists of two convolutional layers, each followed by batch normalization
    and ReLU activation. It is used to extract features from the input at different
    resolutions.
    """

    def __init__(self, in_channels, out_channels, dropout=None, kernel_size=3):
        """
        Initialize the double convolution block.

        Input and output data sizes are nXn, meaning the spatial dimensions
        remain the same after the convolutions. The block consists of two
        convolutional layers with kernel size kernel_size and padding kernel_size // 2, followed by
        batch normalization and ReLU activation. Optionally, dropout can be applied.
        The dropout layer is applied after the second convolution.
        The padding is computed to maintain the spatial dimensions of the input
        data after the convolutions.

        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 1 for grayscale, 3 for RGB, etc.)
        out_channels : int
            Number of output channels after the convolution
        dropout : float, optional
            Dropout rate to apply after the second convolution, by default None.
            If None, no dropout is applied. 
        kernel_size : int, optional
            Size of the convolution kernel, by default 3.
        """
        super(DoubleConv, self).__init__()
        # Validate input parameters
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer") 
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer")
        if dropout is not None and (not isinstance(dropout, (int, float)) or dropout < 0.0):
            raise ValueError("dropout must be a non-negative float or None")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")

        # check that the kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer to maintain spatial dimensions")
        
        #compute the padding size
        padding = kernel_size // 2
        
        # Input data size nXn, output data size nXn
        layers = [
            # Data dimensions: [batch_size, in_channels, height, width] -> [batch_size, out_channels, height, width]
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            # Data dimensions: [batch_size, out_channels, height, width] -> [batch_size, out_channels, height, width]
            nn.BatchNorm2d(out_channels),
            # Data dimensions: [batch_size, out_channels, height, width] -> [batch_size, out_channels, height, width]
            nn.ReLU(inplace=True),
            # Data dimensions: [batch_size, out_channels, height, width] -> [batch_size, out_channels, height, width]
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            # Data dimensions: [batch_size, out_channels, height, width] -> [batch_size, out_channels, height, width]
            nn.BatchNorm2d(out_channels),
            # Data dimensions: [batch_size, out_channels, height, width] -> [batch_size, out_channels, height, width]
            nn.ReLU(inplace=True)
        ]
        
        if dropout is not None and dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
            
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool followed by double conv for the contracting path of UNet.
    
    This block first applies max pooling to reduce the spatial dimensions by half,
    then applies a double convolution block to extract features at the reduced
    resolution. This is used in the encoder path of the UNet.
    """
    
    def __init__(self, in_channels, out_channels, dropout=None, kernel_size=3):
        """
        Initialize the downscaling block.
        
        The input data is first downsampled by a factor of 2 using max pooling,
        then processed through a double convolution block. The spatial dimensions
        are reduced from nXn to (n/2)X(n/2).
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels after the double convolution
        dropout : float, optional
            Dropout rate to apply in the double convolution block, by default None.
            If None, no dropout is applied.
        kernel_size : int, optional
            Size of the convolution kernel used in the double convolution block,
            by default 3. Must be an odd integer to maintain proper padding.
        """
        super(Down, self).__init__()
        
        # Validate input parameters
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer")
        if dropout is not None and (not isinstance(dropout, (int, float)) or dropout < 0.0):
            raise ValueError("dropout must be a non-negative float or None")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer to maintain spatial dimensions")
        
        self.maxpool_conv = nn.Sequential(
            # Data dimensions: [batch_size, in_channels, height, width] -> [batch_size, in_channels, height//2, width//2]
            nn.MaxPool2d(2),
            # Data dimensions: [batch_size, in_channels, height//2, width//2] -> [batch_size, out_channels, height//2, width//2]
            DoubleConv(in_channels, out_channels, dropout, kernel_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the downscaling block.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, out_channels, height//2, width//2]
        """
        # Data dimensions: [batch_size, in_channels, height, width] -> [batch_size, out_channels, height//2, width//2]
        # The max pooling reduces the spatial dimensions by a factor of 2.
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with transposed conv followed by double conv.
    
    This block first upscales the input using either bilinear interpolation or
    transposed convolution, then concatenates it with the corresponding feature
    map from the encoder path (skip connection), and finally applies a double
    convolution block. This is used in the decoder path of the UNet.
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=None, kernel_size=3):
        """
        Initialize the upscaling block for the expansion path of the UNet.
        
        The input data is first upsampled by a factor of 2, then concatenated with
        skip connection features, and processed through a double convolution block.
        The spatial dimensions are increased from nXn to (2n)X(2n).
        
        Parameters
        ----------
        in_channels : int
            Number of input channels from the previous layer
        out_channels : int
            Number of output channels after the double convolution
        bilinear : bool, optional
            Use bilinear upsampling instead of transposed convolution, by default True.
            Bilinear upsampling is generally more stable and uses less memory.
        dropout : float, optional
            Dropout rate to apply in the double convolution block, by default None.
            If None, no dropout is applied.
        kernel_size : int, optional
            Size of the convolution kernel used in the double convolution block,
            by default 3. Must be an odd integer to maintain proper padding.
        """
        super(Up, self).__init__()
        
        # Validate input parameters
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer")
        if not isinstance(bilinear, bool):
            raise ValueError("bilinear must be a boolean")
        if dropout is not None and (not isinstance(dropout, (int, float)) or dropout < 0.0):
            raise ValueError("dropout must be a non-negative float or None")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer to maintain spatial dimensions")
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            # For bilinear: we need to reduce channels manually, then upsample spatially
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),  # Reduce channels
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsample spatially
            )
            # After concatenation: (in_channels//2) + skip_channels
            # The skip connection has the same number of channels as in_channels // 2, so total is in_channels
            # But actually, skip connection has in_channels channels (not in_channels//2)
            # So total is (in_channels//2) + in_channels = 3*in_channels//2
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, dropout, kernel_size)
        else:
            # Transposed convolution: reduces channels and increases spatial size
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After concatenation: (in_channels//2) + skip_channels = in_channels
            # Skip connection has in_channels channels, so total is in_channels//2 + in_channels = 3*in_channels//2
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, dropout, kernel_size)
    
    def forward(self, x1, x2):
        """
        Forward pass through the upscaling block.
        
        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor from the previous decoder layer 
        x2 : torch.Tensor
            Skip connection tensor from the corresponding encoder layer
            
        Returns
        -------
        torch.Tensor
            Output tensor after upsampling and double convolution
        """
        # Upsample x1 to match x2's spatial dimensions
        x1 = self.up(x1)

        # Handle potential size differences between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Pad x1 to match x2's size
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                   diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension: [x2, x1]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_PARROT(L.LightningModule):
    """A UNet architecture for nXn to nXn predictions in PARROT.
    
    This UNet implementation follows the classic encoder-decoder architecture
    with skip connections. It's designed for square input/output data where
    spatial relationships are important.

    Valid input sizes for nXn are any n that is divisible by 16.
    It can also be used for mXn inputs and outputs with proper padding.

    The network downsamples by a factor of 2 at each
    downsampling step, and there are four downsampling steps in total.
    Therefore, the input size must be at least 16x16 to produce a valid output.
    This will require enforced padding in the data to ensure the input size is valid.

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB, etc.)
    num_classes : int
        Number of output classes for classification or 1 for regression
    problem_type : str
        Type of problem to solve, either 'regression' or 'classification'
    batch_size : int
        Size of the batch for training and inference
    bilinear : bool, optional
        Use bilinear upsampling instead of transposed convolution, by default True
    base_channels : int, optional
        Number of channels in the first encoder layer, by default 64
    kernel_size : int, optional
        Size of the convolution kernel used throughout the network, by default 3.
        Must be an odd integer to maintain proper padding.
    **kwargs : dict
        Additional keyword arguments for model configuration, similar to BRNN_PARROT
        - optimizer_name: str, name of the optimizer to use (default: 'AdamW')
        - learn_rate: float, learning rate for the optimizer (default: 1e-3)
        - dropout: float, dropout rate for the model (default: None)
        - monitor: str, metric to monitor for learning rate scheduler (default: 'epoch_val_loss')
        - distributed: bool, whether the model is being trained in a distributed setting (default: False)
        - momentum: float, momentum for SGD optimizer (default: 0.99)
        - beta1: float, first beta parameter for AdamW/Adam optimizer (default: 0.9)
        - beta2: float, second beta parameter for AdamW/Adam optimizer (default: 0.999)
        - eps: float, epsilon for AdamW/Adam optimizer (default: 1e-8)
        - weight_decay: float, weight decay for AdamW/Adam optimizer (default: 1e-2)
    """
    
    def __init__(
        self,
        input_channels,
        num_classes,
        problem_type,
        batch_size,
        bilinear=True,
        base_channels=64,
        kernel_size=3,
        **kwargs
    ):
        super(UNet_PARROT, self).__init__()
        
        # Input validation
        # This checks that the input parameters are valid integers and positive.
        # It also checks that the kernel size is an odd integer to maintain spatial dimensions.
        # The problem type must be either 'regression' or 'classification'.
        if not isinstance(input_channels, int) or input_channels < 1:
            raise ValueError("input_channels must be a positive integer")
        
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        
        if problem_type not in ['regression', 'classification']:
            raise ValueError("problem_type must be either 'regression' or 'classification'")
        
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer to maintain spatial dimensions")
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.problem_type = problem_type
        self.batch_size = batch_size
        self.bilinear = bilinear
        self.base_channels = base_channels
        self.kernel_size = kernel_size
         # Model configuration from kwargs
        self.optimizer_name = kwargs.get("optimizer_name", "AdamW")
        self.learn_rate = kwargs.get("learn_rate", 1e-3)
        self.dropout = kwargs.get("dropout", None)
        self.monitor = kwargs.get("monitor", "epoch_val_loss")
        self.distributed = kwargs.get("distributed", False)
        
        # Handle flexible channel progression
        self.first_down_channels = kwargs.get("first_down_channels", base_channels * 2)
        
        # Validate first_down_channels
        if not isinstance(self.first_down_channels, int) or self.first_down_channels < 1:
            raise ValueError("first_down_channels must be a positive integer")

        # Calculate factor for bilinear upsampling
        factor = 2 if bilinear else 1

        ###
        ### Main UNet architecture
        ###
        ### Data dimensions are mapped throughout
        ### Input data dimensions: (batch_size, input_channels, height, width)
        ### Post downsampling data dimensions: (batch_size, base_channels, height/2, width/2)
        ### Post upsampling data dimensions: (batch_size, base_channels, height, width)
        ### Output data dimensions: (batch_size, num_classes, height, width)

        # Encoder (downsampling path)
        # Data dimensions: (batch_size, input_channels, height, width)
        # Data dimensions after first convolution: (batch_size, base_channels, height, width)
        self.inc = DoubleConv(in_channels=input_channels, out_channels=base_channels, dropout=self.dropout, kernel_size=kernel_size)
        # Data dimensions after first downsampling: (batch_size, first_down_channels, height/2, width/2)
        self.down1 = Down(in_channels=base_channels, out_channels=self.first_down_channels, dropout=self.dropout, kernel_size=kernel_size)
        # Data dimensions after second downsampling: (batch_size, first_down_channels * 2, height/4, width/4)
        self.down2 = Down(in_channels=self.first_down_channels, out_channels=self.first_down_channels * 2, dropout=self.dropout, kernel_size=kernel_size)
        # Data dimensions after third downsampling: (batch_size, first_down_channels * 4, height/8, width/8)
        self.down3 = Down(in_channels=self.first_down_channels * 2, out_channels=self.first_down_channels * 4, dropout=self.dropout, kernel_size=kernel_size)
        # Data dimensions after fourth downsampling: (batch_size, first_down_channels * 8, height/16, width/16)
        self.down4 = Down(in_channels=self.first_down_channels * 4, out_channels=self.first_down_channels * 8 // factor, dropout=self.dropout, kernel_size=kernel_size)

        # Decoder (upsampling path)
        # Data dimensions after first upsampling: (batch_size, first_down_channels * 4, height/8, width/8)
        self.up1 = Up(in_channels=self.first_down_channels * 8 // factor, out_channels=self.first_down_channels * 4 // factor, bilinear=bilinear, dropout=self.dropout, kernel_size=kernel_size)
        # Data dimensions after second upsampling: (batch_size, first_down_channels * 2, height/4, width/4)
        self.up2 = Up(in_channels=self.first_down_channels * 4 // factor, out_channels=self.first_down_channels * 2 // factor, bilinear=bilinear, dropout=self.dropout, kernel_size=kernel_size)
        # Data dimensions after third upsampling: (batch_size, first_down_channels, height/2, width/2)
        self.up3 = Up(in_channels=self.first_down_channels * 2 // factor, out_channels=self.first_down_channels // factor, bilinear=bilinear, dropout=self.dropout, kernel_size=kernel_size)
        # Data dimensions after fourth upsampling: (batch_size, base_channels, height, width)
        self.up4 = Up(in_channels=self.first_down_channels // factor, out_channels=base_channels, bilinear=bilinear, dropout=self.dropout, kernel_size=kernel_size)

        # Output layer - this layer reduces the number of channels to num_classes (1 for regression, num_classes for classification)
        # Data dimensions after output layer: (batch_size, num_classes, height, width)
        # The kernel size is 1 to reduce the number of channels to num_classes while keeping the spatial dimensions the same.
        self.outc = nn.Conv2d(in_channels=base_channels, out_channels=num_classes, kernel_size=1)



        # Set optimizer parameters
        # Stochastic Gradient Descent (SGD) or AdamW
        if self.optimizer_name == "SGD":
            # The momentum parameter is used to accelerate SGD in the relevant direction
            # and dampens oscillations. It is typically set to a value between 0.1 and 0.99.
            self.momentum = kwargs.get("momentum", 0.99)
        elif self.optimizer_name == "AdamW":
            # The beta1 and beta2 parameters are used to control the exponential decay rates
            # for the first and second moment estimates, respectively. They are typically set to
            # values between 0.8 and 0.999.
            self.beta1 = kwargs.get("beta1", 0.9)
            self.beta2 = kwargs.get("beta2", 0.999)
            # The eps parameter is a small constant added to the denominator to prevent division by zero.
            # It is typically set to a value like 1e-8.
            self.eps = kwargs.get("eps", 1e-8)
            # The weight_decay parameter is used to apply L2 regularization to the model parameters.
            # It is typically set to a small value like 1e-2.
            # This helps prevent overfitting by penalizing large weights.
            # It is not used in SGD, but is commonly used in AdamW.
            # If not specified, it defaults to 1e-2.
            self.weight_decay = kwargs.get("weight_decay", 1e-2)
        else:
            raise ValueError("Invalid optimizer name. Supported options: 'SGD', 'AdamW'.")
        
        # Set direction for LR scheduler
        # This is used to determine whether to minimize or maximize the learning rate
        # upon plateau in the learning rate scheduler.
        # The default is to minimize the learning rate.
        direction_map = {"minimize": "min", "maximize": "max"}
        if kwargs.get("direction"):
            self.lr_direction = direction_map[kwargs.get("direction")]
        else:
            self.lr_direction = "min"
        
        # Set loss criteria
        # Loss criteria depend on the problem type.
        # For regression, we use Mean Squared Error (MSE) loss.
        # For classification, we use Cross Entropy loss.
        # Additionally, we set up various classification metrics if the problem is classification.
        # Regression
        if self.problem_type == "regression":
            # R2 score is used to evaluate the performance of regression models.
            # It is a measure of how well the model explains the variance in the target variable.
            # It ranges from 0 to 1, where 1 indicates perfect prediction.
            self.r2_score = R2Score(compute_on_cpu=True)
            # MSE formula: MSE = 1/n * Σ(y_i - ŷ_i)²
            # where y_i is the true value, ŷ_i is the predicted value, and n is the number of samples.
            self.criterion = nn.MSELoss(reduction="mean")
        # Classification
        elif self.problem_type == "classification":
            # The task is set to multiclass classification.
            # This is used to determine the type of classification metrics to compute.
            self.task = "multiclass"
            # Cross Entropy Loss is used for multi-class classification problems.
            # It combines softmax activation and negative log likelihood loss.
            # It is defined as:
            # CrossEntropyLoss = -1/n * Σ(y_i * log(ŷ_i))
            # where y_i is the true class label (one-hot encoded) and ŷ_i is the predicted probability for class i.
            # The reduction is set to "mean" to average the loss over all samples in the batch.
            # This means that the loss is averaged across all samples in the batch,
            # rather than summing them up.
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
            # Set up classification metrics
            # These metrics are used to evaluate the performance of the classification model.
            # They are computed on the validation and test sets.
            # Accuracy: The proportion of correct predictions out of the total predictions.
            # Precision: The proportion of true positive predictions out of all positive predictions.
            # AUROC: Area Under the Receiver Operating Characteristic curve.
            # MCC: Matthews Correlation Coefficient, a measure of the quality of binary classifications.
            # F1 Score: The harmonic mean of precision and recall.
            self.accuracy = Accuracy(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.precision = Precision(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.auroc = AUROC(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.mcc = MatthewsCorrCoef(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
            self.f1_score = F1Score(
                task=self.task, num_classes=self.num_classes, compute_on_cpu=True
            )
        
        # Training loss metric
        # This metric is used to track the training loss during training.
        # It is reset at the end of each epoch.
        # It is used to compute the average training loss over the epoch.
        # It is initialized to zero at the start of training.
        self.train_loss_metric = MeanMetric()
        
        # Save hyperparameters
        # This saves the hyperparameters of the model to the checkpoint.
        # It is useful for logging and reproducibility.
        # It saves the model configuration, optimizer parameters, and other relevant information.
        self.save_hyperparameters()
    
    def forward(self, x):
        """Forward pass through the UNet.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_channels, height, width]
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, num_classes, height, width]
        """
        # Encoder path - downsampling and feature extraction
        # Encoder input data dimensions: [batch_size, input_channels, height, width]
        # Encoder output data dimensions: [batch_size, base_channels, height/16, width/16]
        # Double convolution
        x1 = self.inc(x)
        # Downsampling by a factor of 2 at each step
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections - upsampling and feature fusion
        # Decoder input data dimensions: [batch_size, base_channels * 16 // factor, height/16, width/16]
        # Decoder output data dimensions: [batch_size, base_channels, height, width]
        # Upsampling by a factor of 2 at each step, concatenating with corresponding encoder features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer - final prediction
        # Output layer input data dimensions: [batch_size, base_channels, height, width]
        # Output layer output data dimensions: [batch_size, num_classes, height, width]
        # The output layer reduces the number of channels to num_classes while keeping the spatial dimensions the same.
        # This is done using a 1x1 convolution.
        logits = self.outc(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step for the UNet model."""
        # The data loader should have taken care of padding the input data to ensure
        # that the input size is valid (nXn where n is divisible by 16).
        # The batch should contain inputs and targets.
        inputs, targets = batch
        outputs = self.forward(inputs.float())
        
        if self.problem_type == "regression":
            # floats are used for regression problems.
            loss = self.criterion(outputs, targets.float())
        else:
            # long is used for classification problems.
            loss = self.criterion(outputs, targets.long())
        
        # Training loss metric is used to track the average loss over the epoch.
        self.train_loss_metric(loss)
        # Log the training loss
        self.log("train_loss", loss)
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Compute the mean training loss for the epoch
        epoch_mean = self.train_loss_metric.compute()
        # Log the mean training loss
        self.log("epoch_train_loss", epoch_mean, prog_bar=True, sync_dist=self.distributed)
        # Reset the training loss metric for the next epoch (0)
        self.train_loss_metric.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step for the UNet model."""
        # Validation is run on the validation set to evaluate the model's performance.
        inputs, targets = batch
        outputs = self.forward(inputs.float())
        
        if self.problem_type == "regression":
            loss = self.criterion(outputs, targets.float())
            # Compute R² score if we have enough samples
            if outputs.numel() >= 2:
                # R² score is computed to evaluate the performance of regression models.
                self.r2_score(outputs.flatten(), targets.float().flatten())
                # Log the R² score
                self.log("epoch_val_rsquare", self.r2_score)
        else:
            loss = self.criterion(outputs, targets.long())
            
            # Compute classification metrics and log them
            # Accuracy is the proportion of correct predictions out of the total predictions.
            accuracy = self.accuracy(outputs, targets.long())
            self.log("epoch_val_accuracy", accuracy, on_step=True)
            # F1 Score is the harmonic mean of precision and recall.
            f1score = self.f1_score(outputs, targets.long())
            self.log("epoch_val_f1score", f1score, on_step=True)
            # AUROC is the area under the receiver operating characteristic curve.
            auroc = self.auroc(outputs, targets.long())
            self.log("epoch_val_auroc", auroc, on_step=True)
            # Precision is the proportion of true positive predictions out of all positive predictions.
            precision = self.precision(outputs, targets.long())
            self.log("epoch_val_precision", precision, on_step=True)
            # Matthews Correlation Coefficient (MCC) is a measure of the quality of binary classifications.
            mcc = self.mcc(outputs, targets.long())
            self.log("epoch_val_mcc", mcc, on_step=True)
        # Log the validation loss
        self.log("epoch_val_loss", loss, prog_bar=True, sync_dist=self.distributed)
        self.log("val_loss", loss, sync_dist=self.distributed)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for the UNet model."""
        inputs, targets = batch
        outputs = self.forward(inputs.float())
        
        if self.problem_type == "regression":
            loss = self.criterion(outputs, targets.float())
            # Compute R² score if we have enough samples
            if outputs.numel() >= 2:
                self.r2_score(outputs.flatten(), targets.float().flatten())
                self.log("test_r2_score", self.r2_score)
        else:
            loss = self.criterion(outputs, targets.long())
            accuracy = self.accuracy(outputs, targets.long())
            self.log("test_accuracy", accuracy)
        
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        if self.optimizer_name == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.learn_rate,
                momentum=self.momentum,
                nesterov=True, # Nesterov momentum is used for faster convergence
            )
        elif self.optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learn_rate,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError("Invalid optimizer name. Supported options: 'SGD', 'AdamW'.")
        
        # Learning rate scheduler
        # Cosine Annealing LR scheduler is used to adjust the learning rate during training.
        # It reduces the learning rate following a cosine function, which helps in
        # achieving better convergence and avoiding local minima.
        lr_scheduler = {
            # T_max is the maximum number of epochs for the cosine annealing schedule.
            # eta_min is the minimum learning rate to which the learning rate can decay.
            "scheduler": CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=0.0001
            ),
            "monitor": self.monitor,
            "interval": "epoch",
        }
        
        return [optimizer], [lr_scheduler]