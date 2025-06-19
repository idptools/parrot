# PARROT Base Predictor Architecture

This document describes the new base predictor architecture for PARROT neural networks, which provides a unified and extensible framework for creating predictors that work with any PARROT network.

## Overview

The base predictor architecture consists of several key components:

1. **BasePredictor**: Abstract base class that defines the common interface
2. **LegacyBRNNPredictor**: Handles old-style PARROT model files
3. **LightningPredictor**: Handles PyTorch Lightning checkpoint files
4. **Factory Function**: Automatically detects model type and creates appropriate predictor

## Key Features

- **Unified Interface**: All predictors share the same methods for prediction
- **Automatic Type Detection**: Factory function automatically determines model type
- **Extensible Design**: Easy to create custom predictors for specialized use cases
- **Efficient Batch Processing**: Built-in support for optimized batch predictions with automatic batch size detection
- **Variable Length Support**: Handles sequences of different lengths with automatic padding
- **File I/O**: Direct prediction from/to files with flexible formatting
- **Backward Compatibility**: Existing code continues to work unchanged
- **Device Management**: Automatic CPU/GPU detection and management
- **Input Validation**: Comprehensive validation of sequences and parameters
- **Smart Batch Size Detection**: Automatically detects and uses training batch size for optimal performance

## Basic Usage

### Automatic Model Type Detection (Recommended)

```python
from parrot.base_predictor import create_predictor

# Automatically detect model type and create appropriate predictor
predictor = create_predictor('path/to/model.pt', datatype='sequence')  # datatype only needed for legacy models

# Make predictions
sequence = "ACDEFGHIKLMNPQRSTVWY"
prediction = predictor.predict(sequence)
print(f"Prediction: {prediction}")
```

### Using Specific Predictor Classes

```python
from parrot.base_predictor import LegacyBRNNPredictor, LightningPredictor

# For legacy BRNN models
legacy_predictor = LegacyBRNNPredictor('legacy_model.pt')
legacy_predictor.datatype = 'sequence'  # Must specify for legacy models
legacy_predictor._initialize_network()

# For Lightning models
lightning_predictor = LightningPredictor('lightning_model.ckpt')
```

### Batch Predictions

```python
# Predict multiple sequences at once (uses efficient batching by default)
sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "GGGGGGGGGGGGGGGGGGGG", 
    "PPPPPPPPPPPPPPPPPPPP",
    "AAAAAAAAAAAAAAAAAAAA"
]

# Efficient batch processing (recommended)
predictions = predictor.predict_batch(sequences)
for seq, pred in zip(sequences, predictions):
    print(f"{seq}: {pred}")

# Control batch size manually
predictions = predictor.predict_batch(sequences, batch_size=16)

# Use legacy one-by-one processing if needed
predictions = predictor.predict_batch(sequences, use_efficient_batching=False)

# For very large datasets, use efficient batch processing
large_sequences = ["SEQUENCE" + str(i) for i in range(1000)]
predictions = predictor.predict_batch_efficient(large_sequences, batch_size=32)
```
```

### File-based Predictions

```python
# Predict from file and save results
predictions = predictor.predict_from_file(
    input_file='sequences.txt',      # Format: "seq_id\tsequence" or just "sequence"
    output_file='predictions.txt',
    exclude_seq_id=False,           # Set True if file contains only sequences
    use_efficient_batching=True,    # Use optimized batch processing
    batch_size=32                   # Optional: specify batch size
)

# Example input file formats:
# With sequence IDs (exclude_seq_id=False):
# seq1\tACDEFGHIKLMNPQRSTVWY
# seq2\tGGGGGGGGGGGGGGGGGGGG

# Without sequence IDs (exclude_seq_id=True):
# ACDEFGHIKLMNPQRSTVWY
# GGGGGGGGGGGGGGGGGGGG

# Process large files efficiently
large_predictions = predictor.predict_from_file(
    input_file='large_dataset.txt',
    output_file='large_predictions.txt',
    use_efficient_batching=True,
    batch_size=64  # Larger batch for better GPU utilization
)
```

### Model Information

```python
# Get detailed information about the loaded model
info = predictor.get_model_info()
print("Model Information:")
for key, value in info.items():
    print(f"  {key}: {value}")
```

## Creating Custom Predictors

You can create custom predictors by inheriting from `BasePredictor` and implementing the abstract methods:

```python
from parrot.base_predictor import BasePredictor
import torch.nn as nn

class CustomPredictor(BasePredictor):
    """Custom predictor for specialized use cases."""
    
    def _extract_hyperparameters(self):
        """Extract hyperparameters from your custom model format."""
        # Implement custom parameter extraction logic
        self.hyperparameters = {
            'custom_param1': 'value1',
            'custom_param2': 'value2',
        }
        
        # Set required attributes
        self.datatype = 'sequence'  # or 'residues'
        self.problem_type = 'regression'  # or 'classification'
        self.num_classes = 1
        
        # Setup encoding scheme
        self._setup_encoding('onehot')
    
    def _initialize_network(self):
        """Initialize your custom network architecture."""
        # Create and configure your custom network
        self.network = YourCustomNetwork(**self.hyperparameters)
        self.network.load_state_dict(self.checkpoint)
        self.network.to(self.device)
        self.network.eval()
    
    def _postprocess_prediction(self, raw_output, sequence):
        """Apply custom postprocessing to predictions."""
        prediction = raw_output.detach().cpu().numpy()
        
        # Apply custom transformations
        processed_prediction = self._apply_custom_transforms(prediction)
        
        return processed_prediction
    
    def _apply_custom_transforms(self, prediction):
        """Apply domain-specific transformations."""
        # Example: custom scaling, normalization, etc.
        return prediction * self.custom_scale_factor

# Use your custom predictor
custom_predictor = CustomPredictor('path/to/custom_model.pt')
prediction = custom_predictor.predict('ACDEFGHIKLMNPQRSTVWY')
```

## Backward Compatibility

The existing `py_predictor.Predictor` class continues to work exactly as before:

```python
# This still works unchanged
from parrot import py_predictor
predictor = py_predictor.Predictor('/path/to/model.pt', dtype='sequence')
prediction = predictor.predict('ACDEFGHIKLMNPQRSTVWY')
```

However, the new `create_predictor_auto` function is also available in `py_predictor` for convenience:

```python
from parrot import py_predictor

# New automatic detection function
predictor = py_predictor.create_predictor_auto('/path/to/model.pt', datatype='sequence')
prediction = predictor.predict('ACDEFGHIKLMNPQRSTVWY')
```

## Advanced Features

### Efficient Batch Processing

The base predictor includes sophisticated batch processing that significantly improves performance:

```python
# Automatic batch size detection from training
predictor = create_predictor('model.pt', datatype='sequence')
print(f"Detected batch size: {predictor.batch_size}")

# The predictor automatically uses the training batch size for optimal performance
sequences = ["SEQUENCE" + str(i) for i in range(100)]
predictions = predictor.predict_batch(sequences)  # Uses detected batch size

# Override batch size if needed
predictions = predictor.predict_batch(sequences, batch_size=16)

# Handle variable-length sequences automatically
mixed_length_seqs = [
    "ACDEFGHIKLMNPQRSTVWY",      # Length 20
    "GGGGGGGGGGGG",              # Length 12  
    "PPPPPPPPPPPPPPPPPPPPPPPP"   # Length 24
]
predictions = predictor.predict_batch(mixed_length_seqs)  # Automatic padding
```

### Variable Length Sequence Support

```python
# The predictor handles sequences of different lengths automatically
sequences = [
    "SHORT",                     # Short sequence
    "MEDIUMLENGTHSEQUENCE",      # Medium sequence
    "VERYLONGSEQUENCEWITHMANYRESIDUES"  # Long sequence
]

# Sequences are automatically padded to the same length for efficient batch processing
predictions = predictor.predict_batch(sequences)
# Each prediction will have the correct length for its original sequence
```

### Device Management

```python
# Force CPU usage
predictor = create_predictor('model.pt', datatype='sequence', force_cpu=True)

# Specify device explicitly
predictor = create_predictor('model.pt', datatype='sequence', device='cuda')

# Automatic device detection (default)
predictor = create_predictor('model.pt', datatype='sequence', device='auto')
```

### Input Validation

The base predictor automatically validates input sequences:

- Checks for valid amino acid characters
- Converts to uppercase
- Validates sequence length
- Provides clear error messages for invalid inputs

### Error Handling

The base predictor provides comprehensive error handling:

- Model file validation
- Device availability checking
- Sequence validation
- Clear error messages with context

## Migration Guide

### From Legacy Predictor Code

**Old code:**
```python
from parrot import py_predictor
predictor = py_predictor.Predictor('/path/to/model.pt', dtype='sequence')
prediction = predictor.predict('ACDEFGHIKLMNPQRSTVWY')
```

**New code (recommended):**
```python
from parrot.base_predictor import create_predictor
predictor = create_predictor('/path/to/model.pt', datatype='sequence')
prediction = predictor.predict('ACDEFGHIKLMNPQRSTVWY')
```

### Benefits of Migration

1. **Automatic type detection**: No need to know model format
2. **Better error handling**: More informative error messages
3. **Additional features**: Batch processing, file I/O, model info
4. **Future-proof**: Will support new model formats automatically
5. **Extensible**: Easy to customize for specific needs

## Architecture Details

### Class Hierarchy

```
BasePredictor (Abstract)
├── LegacyBRNNPredictor (for legacy .pt models)
├── LightningPredictor (for PyTorch Lightning .ckpt models)
└── CustomPredictor (User-defined extensions)
```

### Technical Implementation Details

#### Batch Processing Optimization
- **Automatic batch size detection**: Extracts training batch size from model hyperparameters
- **Variable length padding**: Automatically pads sequences to the same length for efficient batching
- **Memory-efficient processing**: Processes large datasets in chunks to avoid memory issues
- **GPU optimization**: Uses larger batch sizes when GPU is available

#### Sequence Encoding
- **Flexible encoding schemes**: Support for one-hot encoding with extensibility for others
- **Batch encoding**: Efficiently encodes multiple sequences simultaneously
- **Automatic padding**: Handles variable-length sequences with zero-padding
- **Device management**: Automatically moves tensors to appropriate device

#### Model Type Detection
- **Automatic detection**: Distinguishes between Legacy and Lightning models
- **Hyperparameter extraction**: Intelligently extracts model configuration
- **State dict handling**: Properly handles different checkpoint formats
- **Error recovery**: Robust error handling for corrupted or incompatible models

### Abstract Methods

All predictor implementations must implement:

1. `_extract_hyperparameters()`: Extract model parameters from checkpoint
2. `_initialize_network()`: Create and configure the network
3. `_postprocess_prediction()`: Process raw model output

### Common Methods (Inherited)

All predictors inherit these methods:

- `predict(sequence)`: Single sequence prediction
- `predict_batch(sequences, use_efficient_batching=True, batch_size=None)`: Batch predictions with optimization options
- `predict_batch_efficient(sequences, batch_size=None)`: Optimized batch processing for large datasets
- `predict_from_file(input_file, output_file=None, exclude_seq_id=False, use_efficient_batching=True, batch_size=None)`: File-based predictions
- `get_model_info()`: Model information including detected batch size
- `_validate_sequence(sequence)`: Input validation
- `_encode_sequence(sequence)`: Single sequence encoding
- `_encode_sequences_batch(sequences)`: Batch sequence encoding with padding

## Best Practices

1. **Use the factory function**: `create_predictor()` for automatic type detection
2. **Leverage efficient batching**: Use `predict_batch()` with default settings for optimal performance
3. **Validate inputs**: The base class does this automatically
4. **Handle errors gracefully**: Use try-catch blocks for file operations
5. **Process large datasets efficiently**: Use `predict_batch_efficient()` or file-based processing
6. **Let auto-detection work**: The predictor automatically detects optimal batch sizes
7. **Specify device appropriately**: Use `device='auto'` or explicit device specification
8. **Handle variable lengths**: The predictor automatically handles sequences of different lengths
9. **Document custom predictors**: Provide clear documentation for custom implementations
10. **Use appropriate batch sizes**: Larger batches for GPU, smaller for CPU or memory constraints

## Performance Examples

### Efficient Large Dataset Processing

```python
from parrot.base_predictor import create_predictor

# Load model
predictor = create_predictor('model.pt', datatype='sequence')

# Process large dataset efficiently
large_sequences = []
with open('large_dataset.txt', 'r') as f:
    for line in f:
        seq_id, sequence = line.strip().split('\t')
        large_sequences.append(sequence)

print(f"Processing {len(large_sequences)} sequences...")

# Use efficient batch processing
predictions = predictor.predict_batch_efficient(
    large_sequences, 
    batch_size=64  # Optimize based on your GPU memory
)

# Save results
with open('predictions.txt', 'w') as f:
    for i, pred in enumerate(predictions):
        f.write(f"seq_{i}\t{pred}\n")

print("Processing complete!")
```

### Memory-Efficient File Processing

```python
# For very large files, use direct file processing
# This processes the file in chunks without loading everything into memory
predictions = predictor.predict_from_file(
    input_file='huge_dataset.txt',
    output_file='predictions.txt',
    use_efficient_batching=True,
    batch_size=32  # Smaller batch size for memory efficiency
)

print(f"Processed {len(predictions)} sequences")
```

### Variable Length Sequence Handling

```python
# Mixed length sequences are handled automatically
sequences = [
    "MKLL",                                    # 4 residues
    "ACDEFGHIKLMNPQRSTVWY",                   # 20 residues  
    "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",    # 35 residues
    "A"                                        # 1 residue
]

# All sequences processed efficiently in one batch
predictions = predictor.predict_batch(sequences)

# Each prediction corresponds to its original sequence length
for seq, pred in zip(sequences, predictions):
    print(f"Sequence length {len(seq)}: prediction shape {pred.shape}")
```

## Troubleshooting

### Common Issues

1. **Model file not found**: Check file path and permissions
2. **CUDA not available**: Set `force_cpu=True` or `device='cpu'`
3. **Invalid sequences**: Check for non-amino acid characters
4. **Legacy model datatype**: Specify `datatype` parameter for old models
5. **Memory issues**: Use smaller batch sizes or efficient batching for large datasets
6. **Variable sequence lengths**: The predictor handles this automatically with padding
7. **Batch size optimization**: Let the predictor auto-detect optimal batch size
8. **File format issues**: Ensure proper tab-separated format for sequence files
9. **Import errors**: Ensure all required dependencies (torch, numpy) are installed

### Getting Help

- Check the examples in `predictor_examples.py`
- Review error messages carefully - they provide specific guidance
- Ensure model files are accessible and not corrupted
- Verify that sequences contain only valid amino acid characters

## Future Enhancements

Planned features for future versions:

1. **Additional encoding schemes**: Support for custom encodings beyond one-hot

## Current Implementation Notes

### Known Limitations
- Error handling could be enhanced in certain edge cases

### Dependencies
The base predictor requires:
- PyTorch
- NumPy  
- PyTorch Lightning (for Lightning model support)
- PARROT core modules (`brnn_architecture`, `encode_sequence`, `validate_args`)

Make sure all dependencies are properly installed for full functionality.
