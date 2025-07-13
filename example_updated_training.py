#!/usr/bin/env python3
"""
Example demonstrating the updated train_network.py functionality with UNet and cross-validation.

This example shows how to:
1. Use the updated train_network.py with UNet architectures
2. Handle different datatypes (sequence, residues, matrix)
3. Perform cross-validation with Lightning
4. Test models with the updated test functions

Author: Updated PARROT training system
"""

import os
import sys
import tempfile
import torch
import numpy as np
from omegaconf import DictConfig

# Add the parrot package to the path
sys.path.insert(0, '/home/nrazo/packages/parrot')

from parrot.train_network import train, test_labeled_data
from parrot.unet_architecture import UNet_PARROT
from parrot.brnn_architecture import BRNN_PARROT
from parrot.process_input_data import SequenceDataset
from parrot.encode_sequence import ParrotLightningEncoder


def create_sample_data():
    """Create sample data files for demonstration"""
    
    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    print(f"Creating sample data in: {tmpdir}")
    
    # Create sequence-level regression data
    seq_reg_file = os.path.join(tmpdir, 'sequence_regression.tsv')
    with open(seq_reg_file, 'w') as f:
        f.write("# Sequence-level regression data\n")
        f.write("# seqID\tsequence\ttarget_value\n")
        for i in range(50):
            seq = "ACDEFGHIKLMNPQRSTVWY"[:10 + i % 10]
            value = np.random.random() * 10
            f.write(f"seq_{i}\t{seq}\t{value:.4f}\n")
    
    # Create residue-level classification data
    res_class_file = os.path.join(tmpdir, 'residue_classification.tsv')
    with open(res_class_file, 'w') as f:
        f.write("# Residue-level classification data\n")
        f.write("# seqID\tsequence\tclass_labels...\n")
        for i in range(50):
            seq = "ACDEFGHIKLMNPQRSTVWY"[:8 + i % 5]
            classes = [str(np.random.randint(0, 3)) for _ in range(len(seq))]
            f.write(f"seq_{i}\t{seq}\t{' '.join(classes)}\n")
    
    # Create multi-column sequence data (simulating matrix-like data)
    matrix_file = os.path.join(tmpdir, 'matrix_data.tsv')
    with open(matrix_file, 'w') as f:
        f.write("# Multi-column sequence data\n")
        f.write("# seqID\tseq_part1\tseq_part2\ttarget_values...\n")
        for i in range(50):
            seq1 = "ACDEFGHIKLMNPQRSTVWY"[:5 + i % 3]
            seq2 = "ACDEFGHIKLMNPQRSTVWY"[:5 + (i+1) % 3]
            total_len = len(seq1) + len(seq2) + 1  # +1 for delimiter
            values = [f"{np.random.random():.4f}" for _ in range(total_len)]
            f.write(f"seq_{i}\t{seq1}\t{seq2}\t{' '.join(values)}\n")
    
    return tmpdir, seq_reg_file, res_class_file, matrix_file


def example_traditional_brnn_training():
    """Example of traditional BRNN training with the updated system"""
    print("\n=== Example 1: Traditional BRNN Training ===")
    
    tmpdir, seq_reg_file, res_class_file, matrix_file = create_sample_data()
    
    try:
        # Create encoder configuration
        encoder_cfg = DictConfig({
            'type': 'table',
            'alphabet': 'ACDEFGHIKLMNPQRSTVWY*'
        })
        
        # Create datasets
        train_dataset = SequenceDataset(seq_reg_file, encoder_cfg=encoder_cfg)
        val_dataset = SequenceDataset(seq_reg_file, encoder_cfg=encoder_cfg)  # Same for demo
        
        print(f"Dataset type detected: {train_dataset.datatype}")
        print(f"Number of training samples: {len(train_dataset)}")
        
        # Create a simple BRNN for testing
        class SimpleBRNN(torch.nn.Module):
            def __init__(self, input_size, hidden_size=64, output_size=1):
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
                self.fc = torch.nn.Linear(hidden_size * 2, output_size)
            
            def forward(self, x):
                # x: (batch, seq_len, input_size)
                output, _ = self.rnn(x)
                # For sequence-level prediction, use the last output
                last_output = output[:, -1, :]  # (batch, hidden_size * 2)
                return self.fc(last_output)  # (batch, output_size)
        
        # Get encoding size from a sample
        sample_seq = "ACDEFG"
        encoder = ParrotLightningEncoder(encoder_cfg)
        encoded = encoder.encode(sample_seq)
        input_size = encoded.shape[1]
        
        brnn = SimpleBRNN(input_size)
        
        # Train the model
        weights_file = os.path.join(tmpdir, 'brnn_weights.pt')
        print("Training BRNN...")
        
        train_losses, val_losses = train(
            network=brnn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            datatype='sequence',
            problem_type='regression',
            weights_file=weights_file,
            stop_condition='iter',
            device='cpu',
            learn_rate=0.001,
            n_epochs=5,
            batch_size=8,
            verbose=True,
            silent=False
        )
        
        print(f"Training completed! Final train loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        
        # Test the trained model
        print("Testing trained model...")
        test_loss, predictions = test_labeled_data(
            network=brnn,
            test_dataset=val_dataset,
            datatype='sequence',
            problem_type='regression',
            weights_file=weights_file,
            num_classes=1,
            probabilistic_classification=False,
            include_figs=False,
            device='cpu',
            batch_size=8
        )
        
        print(f"Test loss: {test_loss:.4f}")
        print(f"Number of predictions: {len(predictions)}")
        
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def example_unet_training():
    """Example of UNet training with the updated system"""
    print("\n=== Example 2: UNet Training Setup ===")
    
    # Create a UNet model
    unet = UNet_PARROT(
        input_channels=1,
        num_classes=3,  # 3-class classification
        problem_type='classification',
        batch_size=4,
        base_channels=32,
        first_down_channels=64,
        learn_rate=0.001,
        optimizer_name='AdamW'
    )
    
    print(f"Created UNet with {unet.input_channels} input channels")
    print(f"Problem type: {unet.problem_type}")
    print(f"Number of classes: {unet.num_classes}")
    
    # Test forward pass
    test_input = torch.randn(4, 1, 64, 64)  # Batch of 4, 1 channel, 64x64
    with torch.no_grad():
        output = unet(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # For actual training, you would need matrix-format data
    # This demonstrates the setup without actual training due to data format complexity
    print("UNet training would require matrix-format input data")
    print("The train() function will automatically detect UNet and use Lightning training")


def example_cross_validation():
    """Example of cross-validation with Lightning"""
    print("\n=== Example 3: Cross-Validation Setup ===")
    
    tmpdir, seq_reg_file, res_class_file, matrix_file = create_sample_data()
    
    try:
        # Create encoder configuration
        encoder_cfg = DictConfig({
            'type': 'table',
            'alphabet': 'ACDEFGHIKLMNPQRSTVWY*'
        })
        
        # Create dataset
        full_dataset = SequenceDataset(seq_reg_file, encoder_cfg=encoder_cfg)
        
        # Create a UNet for cross-validation
        unet = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=4,
            base_channels=16,  # Small for demo
            first_down_channels=32,
            learn_rate=0.001
        )
        
        weights_file = os.path.join(tmpdir, 'cv_weights.pt')
        
        print("Setting up cross-validation...")
        print(f"Dataset size: {len(full_dataset)}")
        print("Cross-validation would use Lightning's built-in functionality")
        
        # Demonstrate the cross-validation call (without actual execution due to data format)
        print("\nCross-validation call would be:")
        print("""
cv_results = train(
    network=unet,
    train_dataset=full_dataset,
    val_dataset=None,  # Will be split automatically
    datatype='matrix',  # UNet requires matrix data
    problem_type='regression',
    weights_file=weights_file,
    stop_condition='iter',
    device='cpu',
    learn_rate=0.001,
    n_epochs=10,
    cross_validation=True,
    cv_folds=5,
    batch_size=4
)
        """)
        
        print("This would return a dictionary with fold results and statistics")
        
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def example_datatype_detection():
    """Example of automatic datatype detection"""
    print("\n=== Example 4: Automatic Datatype Detection ===")
    
    tmpdir, seq_reg_file, res_class_file, matrix_file = create_sample_data()
    
    try:
        encoder_cfg = DictConfig({
            'type': 'table',
            'alphabet': 'ACDEFGHIKLMNPQRSTVWY*'
        })
        
        # Test different file types
        datasets = [
            ("Sequence regression", seq_reg_file),
            ("Residue classification", res_class_file),
            ("Multi-column (matrix-like)", matrix_file)
        ]
        
        for name, filepath in datasets:
            dataset = SequenceDataset(filepath, encoder_cfg=encoder_cfg)
            print(f"{name}: detected as '{dataset.datatype}'")
            print(f"  - Has multi-columns: {dataset.has_multi_columns}")
            print(f"  - Sample count: {len(dataset)}")
            
            # Show a sample
            sample_id, sample_seq, sample_values = dataset.data[0]
            print(f"  - Sample: {sample_id[:10]}... -> {str(sample_values)[:50]}...")
            print()
    
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def main():
    """Run all examples"""
    print("PARROT Updated Training System Examples")
    print("=" * 50)
    
    example_datatype_detection()
    example_traditional_brnn_training()
    example_unet_training()
    example_cross_validation()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nKey features of the updated system:")
    print("1. ✓ Supports both BRNN and UNet architectures")
    print("2. ✓ Automatic network type detection")
    print("3. ✓ Lightning-based training for UNet")
    print("4. ✓ Cross-validation support")
    print("5. ✓ Flexible datatype handling (sequence/residues/matrix)")
    print("6. ✓ Backward compatibility with existing BRNN code")


if __name__ == "__main__":
    main()
