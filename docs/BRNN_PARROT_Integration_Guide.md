# BRNN_PARROT: Unified Many-to-Many and Many-to-One Architecture

## Overview

The `BRNN_PARROT` class is a unified PyTorch Lightning module that integrates the functionality of both `BRNN_MtM` (Many-to-Many) and `BRNN_MtO` (Many-to-One) architectures into a single, flexible class. It uses the `datatype` parameter to automatically switch between the two modes, with additional support from the new `ParrotDataModule` for streamlined data handling.

## Key Features

- **Unified Architecture**: Single class that can handle both MtM and MtO tasks
- **Automatic Mode Selection**: Uses `datatype` to determine architecture behavior
- **Complete Integration**: Handles all differences between MtM and MtO in forward pass, training, validation, and testing
- **Enhanced Architecture**: Includes layer normalization, flexible linear layers, and advanced optimizer support
- **Data Module Integration**: Works seamlessly with `ParrotDataModule` for PyTorch Lightning workflows
- **Legacy Support**: Maintains `BRNN_PARROT_LEGACY` class for backward compatibility

## Architecture Modes

### Many-to-One (MtO) Mode
- **Trigger**: `datatype = "sequence"`
- **Use Case**: Sequence-level predictions (one output per sequence)
- **Examples**: Protein stability, molecular properties, sequence classification
- **Output Shape**: `(batch_size, num_classes)`

### Many-to-Many (MtM) Mode  
- **Trigger**: `datatype = "residues"`
- **Use Case**: Residue-level predictions (one output per residue)
- **Examples**: Secondary structure, disorder, binding sites
- **Output Shape**: `(batch_size, sequence_length, num_classes)`

## Key Integration Points

### 1. Forward Pass Logic

The forward method automatically adapts based on `datatype`:

```python
def forward(self, x):
    # Always run LSTM forward pass
    out, (h_n, c_n) = self.lstm(x)
    
    # Different processing if MtM or MtO
    if self.datatype == "sequence":
        # Many-to-One: use only final hidden states from both directions
        # Retain the outputs of the last time step in the sequence for both directions
        # (i.e. output of seq[n] in forward direction, seq[0] in reverse direction)
        out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
    
    # Apply layer normalization for improved stability
    out = self.layer_norm(out)
    
    # Apply linear layers
    for layer in self.linear_layers:
        out = layer(out)
    return out
```

### 2. Enhanced Architecture Features

The current implementation includes several enhancements:

- **Layer Normalization**: Applied to LSTM outputs for improved training stability
- **Flexible Linear Layers**: Support for multiple linear layers with configurable hidden sizes
- **Advanced Dropout**: Applied to both LSTM and linear layers when specified
- **Multiple Optimizers**: Support for SGD (with Nesterov momentum) and AdamW
- **Learning Rate Scheduling**: Cosine annealing scheduler with configurable monitoring

### 3. Target Handling

The training, validation, and test steps automatically handle target tensor reshaping:

```python
if self.problem_type == "regression":
    # Handle target reshaping for MtO regression
    if self.datatype == 'sequence':
        targets = targets.view(-1, 1)
    loss = self.criterion(outputs, targets.float())
else:
    if self.datatype == "residues":
        outputs = outputs.permute(0, 2, 1)
    loss = self.criterion(outputs, targets.long())
```

### 4. Data Module Integration

The `ParrotDataModule` provides seamless integration with PyTorch Lightning:

```python
# Automatic data loading and splitting
data_module = ParrotDataModule(
    tsv_file="data.tsv",
    num_classes=1,
    datatype="sequence",
    batch_size=32,
    fractions=[0.6, 0.25, 0.15]
)
```

### 5. Output Shapes

| Mode | datatype | Input Shape | Output Shape |
|------|----------|-------------|--------------|
| MtO  | "sequence" | (B, L, F)   | (B, C)       |
| MtM  | "residues" | (B, L, F)   | (B, L, C)    |

Where:
- B = batch_size
- L = sequence_length  
- F = input_features (e.g., 20 for one-hot amino acids)
- C = num_classes

## Usage Examples

### Example 1: Sequence Regression (MtO) with Data Module

```python
from parrot.brnn_architecture import BRNN_PARROT, ParrotDataModule
import pytorch_lightning as pl

# Set up data module
data_module = ParrotDataModule(
    tsv_file="sequence_data.tsv",
    num_classes=1,                   # Regression
    datatype="sequence",             # MtO mode
    batch_size=32,
    fractions=[0.6, 0.25, 0.15]
)

# Create model for sequence-level regression
model = BRNN_PARROT(
    input_size=20,                   # One-hot encoded amino acids
    lstm_hidden_size=128,
    num_lstm_layers=2,
    num_classes=1,                   # Regression
    problem_type="regression",
    datatype="sequence",             # MtO mode
    batch_size=32,
    num_linear_layers=2,             # Multiple linear layers
    linear_hidden_size=64,           # Hidden layer size
    dropout=0.2,                     # Dropout for regularization
    optimizer_name="AdamW",          # Advanced optimizer
    learn_rate=1e-3
)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)

# Input: batch of sequences
input_tensor = torch.randn(32, 100, 20)  # 32 sequences, length 100
# Output: one value per sequence
output = model(input_tensor)  # Shape: (32, 1)
```

### Example 2: Residue Classification (MtM) with Enhanced Features

```python
# Set up data module for residue-level data
data_module = ParrotDataModule(
    tsv_file="residue_data.tsv",
    num_classes=3,                   # 3-class classification
    datatype="residues",             # MtM mode
    batch_size=16,
    encode="onehot"
)

# Create model for residue-level classification
model = BRNN_PARROT(
    input_size=20,
    lstm_hidden_size=64,
    num_lstm_layers=2,
    num_classes=3,                   # 3-class classification
    problem_type="classification",
    datatype="residues",             # MtM mode
    batch_size=16,
    num_linear_layers=1,             # Single output layer
    dropout=0.3,                     # Higher dropout for classification
    optimizer_name="SGD",            # SGD with momentum
    momentum=0.9,
    learn_rate=1e-2
)

# Input: batch of sequences
input_tensor = torch.randn(16, 100, 20)  # 16 sequences, length 100
# Output: classification for each residue
output = model(input_tensor)  # Shape: (16, 100, 3)
```

### Example 3: Advanced Configuration

```python
# Advanced model configuration with all features
model = BRNN_PARROT(
    input_size=20,
    lstm_hidden_size=256,
    num_lstm_layers=3,               # Deeper network
    num_classes=1,
    problem_type="regression",
    datatype="sequence",
    batch_size=64,
    # Linear layer configuration
    num_linear_layers=3,             # Multiple hidden layers
    linear_hidden_size=128,          # Hidden layer size
    dropout=0.25,                    # Regularization
    # Optimizer configuration
    optimizer_name="AdamW",
    learn_rate=5e-4,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=1e-2,
    # Training configuration
    monitor="epoch_val_loss",
    direction="minimize"
)
```

## Architecture Details

### Core Components

1. **LSTM Layer**: Bidirectional LSTM with configurable hidden size and number of layers
2. **Layer Normalization**: Applied to LSTM outputs for improved training stability
3. **Linear Layers**: Flexible number of linear layers with optional hidden layers
4. **Dropout**: Applied to both LSTM (when multiple layers) and linear layers
5. **Activation Functions**: ReLU activation between linear layers

### Enhanced Features

- **Multiple Linear Layers**: Configurable depth with hidden layers
- **Advanced Optimizers**: SGD with Nesterov momentum and AdamW support
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Comprehensive Metrics**: R², accuracy, precision, F1-score, AUROC, MCC
- **Distributed Training**: Support for multi-GPU training

## Integration Analysis Summary

After analyzing the updated `BRNN_PARROT` class, here's the current state:

### ✅ **Current Architecture Features**

1. **Datatype-Based Selection**: Uses `datatype` parameter instead of `num_predicted_values`
2. **Enhanced Architecture**: Layer normalization, flexible linear layers, advanced dropout
3. **Multiple Optimizers**: SGD with Nesterov momentum and AdamW with full parameter control
4. **Comprehensive Metrics**: Extensive metric tracking for both regression and classification
5. **Data Module Integration**: Seamless PyTorch Lightning workflow with `ParrotDataModule`
6. **Distributed Training**: Built-in support for multi-GPU training

### 🔧 **Key Architectural Changes**

1. **Forward Pass Logic**: Based on `datatype == "sequence"` (MtO) vs `datatype == "residues"` (MtM)
2. **Layer Normalization**: Added for improved training stability and performance
3. **Flexible Linear Layers**: Support for multiple hidden layers with configurable sizes
4. **Advanced Dropout**: Applied strategically to both LSTM and linear components
5. **Enhanced Loss Handling**: Proper target reshaping and output permutation for different modes

### 🎯 **Current Capabilities**

- ✅ MtO mode produces correct output shape: `(batch_size, num_classes)`
- ✅ MtM mode produces correct output shape: `(batch_size, seq_len, num_classes)`  
- ✅ Target tensors are handled correctly for both modes
- ✅ Loss calculations work for both regression and classification
- ✅ Enhanced architecture with layer normalization and flexible linear layers
- ✅ Advanced optimizer support with full hyperparameter control
- ✅ Comprehensive metric tracking and logging

## Recommendations

1. **Use BRNN_PARROT for new projects** - the unified architecture with enhanced features
2. **Set `datatype="sequence"`** for sequence-level predictions (MtO mode)
3. **Set `datatype="residues"`** for residue-level predictions (MtM mode)
4. **Use ParrotDataModule** for streamlined PyTorch Lightning workflows
5. **Leverage enhanced features**: multiple linear layers, layer normalization, advanced optimizers
6. **Consider distributed training** for large datasets using the built-in distributed support

## Migration Guide

### From Legacy BRNN Classes:

```python
# Old way (separate classes)
from parrot.brnn_architecture import BRNN_MtO, BRNN_MtM

# Sequence-level prediction
model_mto = BRNN_MtO(datatype="sequence", input_size=20, hidden_size=64, ...)

# Residue-level prediction  
model_mtm = BRNN_MtM(datatype="residues", input_size=20, hidden_size=64, ...)

# New way (unified class)
from parrot.brnn_architecture import BRNN_PARROT

# Sequence-level prediction
model_sequence = BRNN_PARROT(
    datatype="sequence",        # MtO mode
    input_size=20, 
    lstm_hidden_size=64,
    num_lstm_layers=2,
    num_classes=1,
    problem_type="regression",
    batch_size=32
)

# Residue-level prediction
model_residues = BRNN_PARROT(
    datatype="residues",        # MtM mode
    input_size=20,
    lstm_hidden_size=64, 
    num_lstm_layers=2,
    num_classes=3,
    problem_type="classification",
    batch_size=32
)
```

### Using the Data Module:

```python
# Complete PyTorch Lightning workflow
from parrot.brnn_architecture import BRNN_PARROT, ParrotDataModule
import pytorch_lightning as pl

# Set up data
data_module = ParrotDataModule(
    tsv_file="data.tsv",
    num_classes=1,
    datatype="sequence",
    batch_size=32
)

# Set up model
model = BRNN_PARROT(
    input_size=20,
    lstm_hidden_size=128,
    num_lstm_layers=2,
    num_classes=1,
    problem_type="regression",
    datatype="sequence",
    batch_size=32,
    optimizer_name="AdamW",
    learn_rate=1e-3
)

# Train
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)
```

## Conclusion

The `BRNN_PARROT` class successfully provides a unified, enhanced architecture that handles both Many-to-Many and Many-to-One tasks using the `datatype` parameter as the mode selector. The current implementation offers significant improvements over the original separate classes:

### Key Improvements:
- **Enhanced Architecture**: Layer normalization, flexible linear layers, strategic dropout
- **Advanced Optimizers**: Full SGD and AdamW support with comprehensive hyperparameter control
- **Data Integration**: Seamless PyTorch Lightning workflow with `ParrotDataModule`
- **Comprehensive Metrics**: Extensive tracking for both regression and classification tasks
- **Production Ready**: Distributed training support and robust error handling

### Architecture Handling:
- **Forward pass logic**: Automatically switches between sequence outputs (MtM) vs. final hidden states (MtO)
- **Target tensor handling**: Proper reshaping for MtO regression across all training steps
- **Output shapes**: Correct tensor dimensions for both per-sequence and per-residue predictions
- **Loss calculations**: Compatible with both regression and classification in both modes

### Legacy Support:
The `BRNN_PARROT_LEGACY` class remains available for backward compatibility, ensuring existing code continues to work while new projects can leverage the enhanced unified architecture.

This provides a more maintainable, feature-rich, and production-ready solution for bidirectional recurrent neural networks in protein sequence analysis.
