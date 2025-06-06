# Example Usage for PARROT Sequence Regression Optimization

## Using the config file with parrot-optimize

To run hyperparameter optimization using the provided config file:

```bash
# Basic usage with config file
python scripts/parrot-optimize --config data/sequence_regression_config.yaml

# Override specific parameters from command line
python scripts/parrot-optimize \
    --config data/sequence_regression_config.yaml \
    --tsv_file your_dataset.tsv \
    --study_name your_experiment_name \
    --n_trials 100

# Run with custom data splits
python scripts/parrot-optimize \
    --config data/sequence_regression_config.yaml \
    --tsv_file your_dataset.tsv \
    --split_file your_splits.txt \
    --batch_size 128
```

## Config File Explanation

### Key Parameters for Good Performance:

1. **Learning Rate Range (0.0001-0.01)**: Conservative range that works well for most regression tasks
2. **LSTM Architecture**: 1-3 layers with 64-512 hidden units provides good expressiveness
3. **Linear Layers**: 1-4 layers with 32-256 units for final prediction layers
4. **Dropout (0.0-0.4)**: Helps prevent overfitting
5. **AdamW Optimizer**: Generally more stable than SGD for regression
6. **Batch Size 64**: Good balance between training stability and memory usage

### Customization Tips:

- **For larger datasets**: Increase batch_size to 128 or 256
- **For smaller datasets**: Reduce max epochs to 50, increase min_delta to 0.005
- **For more complex sequences**: Increase lstm_hidden_size max to 768
- **For simpler tasks**: Reduce architecture complexity by lowering max values

### Expected Results:

This configuration should find models with:
- Training loss: ~0.5-3.0 (depending on your data scale)
- Validation loss: Within 10-20% of training loss
- R² scores: >0.7 for well-conditioned regression problems

The optimization will try 50 different hyperparameter combinations and should complete in a few hours on modern hardware.
