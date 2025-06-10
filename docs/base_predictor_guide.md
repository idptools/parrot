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
- **Batch Processing**: Built-in support for batch predictions
- **File I/O**: Direct prediction from/to files
- **Backward Compatibility**: Existing code continues to work unchanged
- **Device Management**: Automatic CPU/GPU detection and management
- **Input Validation**: Comprehensive validation of sequences and parameters

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
# Predict multiple sequences at once
sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "GGGGGGGGGGGGGGGGGGGG",
    "PPPPPPPPPPPPPPPPPPPP"
]

predictions = predictor.predict_batch(sequences)
for seq, pred in zip(sequences, predictions):
    print(f"{seq}: {pred}")
```

### File-based Predictions

```python
# Predict from file and save results
predictions = predictor.predict_from_file(
    input_file='sequences.txt',  # Format: "seq_id\tsequence"
    output_file='predictions.txt'
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
├── LegacyBRNNPredictor
├── LightningPredictor
└── CustomPredictor (User-defined)
```

### Abstract Methods

All predictor implementations must implement:

1. `_extract_hyperparameters()`: Extract model parameters from checkpoint
2. `_initialize_network()`: Create and configure the network
3. `_postprocess_prediction()`: Process raw model output

### Common Methods (Inherited)

All predictors inherit these methods:

- `predict(sequence)`: Single sequence prediction
- `predict_batch(sequences)`: Batch predictions
- `predict_from_file(input_file, output_file)`: File-based predictions
- `get_model_info()`: Model information
- `_validate_sequence(sequence)`: Input validation
- `_encode_sequence(sequence)`: Sequence encoding

## Best Practices

1. **Use the factory function**: `create_predictor()` for automatic type detection
2. **Validate inputs**: The base class does this automatically
3. **Handle errors gracefully**: Use try-catch blocks for file operations
4. **Batch when possible**: Use `predict_batch()` for multiple sequences
5. **Specify device**: Use `device='auto'` or explicit device specification
6. **Document custom predictors**: Provide clear documentation for custom implementations

## Examples

See `predictor_examples.py` for comprehensive examples of all features and use cases.

## Troubleshooting

### Common Issues

1. **Model file not found**: Check file path and permissions
2. **CUDA not available**: Set `force_cpu=True` or `device='cpu'`
3. **Invalid sequences**: Check for non-amino acid characters
4. **Legacy model datatype**: Specify `datatype` parameter for old models
5. **Memory issues**: Use batch processing for large datasets

### Getting Help

- Check the examples in `predictor_examples.py`
- Review error messages carefully - they provide specific guidance
- Ensure model files are accessible and not corrupted
- Verify that sequences contain only valid amino acid characters

## Future Enhancements

Planned features for future versions:

1. **Additional encoding schemes**: Support for custom encodings
2. **Streaming predictions**: For very large datasets
3. **Model ensembling**: Combine predictions from multiple models
4. **Uncertainty quantification**: Confidence intervals for predictions
5. **Performance optimization**: Further speed improvements
