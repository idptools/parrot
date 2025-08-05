#!/usr/bin/env python3
"""
Demo script showing how to use the save/load functionality for PARROT encoders.

This script demonstrates:
1. Creating different types of encoders
2. Saving them to files
3. Loading them back
4. Verifying that they work identically
5. Saving configuration files
"""

import tempfile
import torch
from omegaconf import OmegaConf
from pathlib import Path

# Import the encoder classes
from parrot.encode_sequence import (
    ParrotLightningEncoder, 
    TableParrotEncoder, 
    FunctionalParrotEncoder,
    MatrixParrotEncoder,
    BaseParrotEncoder
)

def demo_table_encoder_save_load():
    """Demo save/load for TableParrotEncoder (one-hot encoding from alphabet)."""
    print("=" * 60)
    print("DEMO: TableParrotEncoder Save/Load (One-hot from alphabet)")
    print("=" * 60)
    
    # Create a simple table encoder from alphabet (generates one-hot encoding)
    config = OmegaConf.create({
        "type": "table",
        "alphabet": "ACDEFGHIKLMNPQRSTVWY"  # Standard amino acids
    })
    
    # Create the encoder
    original_encoder = TableParrotEncoder(config)
    print(f"Original encoder alphabet: {sorted(original_encoder.get_alphabet())}")
    print(f"Original encoder input size: {len(original_encoder)}")
    
    # Test with a protein sequence
    test_sequence = "ACDEFGHIK"
    print(f"Test sequence: {test_sequence}")
    
    # Encode the sequence
    encoded = original_encoder.encode(test_sequence)
    print(f"Encoded tensor shape: {encoded.shape}")
    
    # Decode it back
    decoded = original_encoder.decode(encoded)
    print(f"Decoded sequence: {decoded[0]}")
    
    # Save the encoder
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "table_encoder.pkl"
        original_encoder.save(str(save_path))
        print(f"Encoder saved to: {save_path}")
        
        # Load it back
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        print(f"Loaded encoder type: {type(loaded_encoder).__name__}")
        print(f"Loaded encoder alphabet: {sorted(loaded_encoder.get_alphabet())}")
        print(f"Loaded encoder input size: {len(loaded_encoder)}")
        
        # Test that it works identically
        loaded_encoded = loaded_encoder.encode(test_sequence)
        loaded_decoded = loaded_encoder.decode(loaded_encoded)
        
        print(f"Tensors identical: {torch.equal(encoded, loaded_encoded)}")
        print(f"Decoded sequences identical: {decoded == loaded_decoded}")
        
        # Save configuration
        config_path = Path(temp_dir) / "table_config.json"
        original_encoder.save_config(str(config_path))
        print(f"Configuration saved to: {config_path}")
        
        # Show config content
        with open(config_path, 'r') as f:
            print("Config content:")
            print(f.read())

def demo_matrix_encoder_save_load():
    """Demo save/load for MatrixParrotEncoder."""
    print("\n" + "=" * 60)
    print("DEMO: MatrixParrotEncoder Save/Load (Vectorial encoding)")
    print("=" * 60)
    
    # Create a matrix encoder
    config = OmegaConf.create({
        "type": "matrix",
        "alphabet": "ACGT",  # DNA/RNA sequences
        "gap_char": "-",
        "use_gap": True,
        "encoding_type": "vectorial"
    })
    
    original_encoder = MatrixParrotEncoder(config)
    print(f"Original encoder alphabet: {sorted(original_encoder.get_alphabet())}")
    print(f"Original encoder input size: {len(original_encoder)}")
    print(f"Encoding type: {original_encoder.encoding_type}")
    print(f"Uses gap character: {original_encoder.use_gap}")
    
    # Test with a DNA sequence with gap
    test_sequence = "ACGT-A"
    print(f"Test sequence: {test_sequence}")
    
    # Encode the sequence (creates a matrix)
    encoded = original_encoder.encode(test_sequence)
    print(f"Encoded tensor shape: {encoded.shape}")  # Should be (6, 6, input_size)
    
    # Decode it back
    decoded = original_encoder.decode(encoded)
    print(f"Decoded sequence: {decoded[0]}")
    
    # Save and load
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "matrix_encoder.pkl"
        original_encoder.save(str(save_path))
        
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        print(f"Loaded encoder type: {type(loaded_encoder).__name__}")
        
        # Test identity
        loaded_encoded = loaded_encoder.encode(test_sequence)
        loaded_decoded = loaded_encoder.decode(loaded_encoded)
        
        print(f"Tensors identical: {torch.equal(encoded, loaded_encoded)}")
        print(f"Decoded sequences identical: {decoded == loaded_decoded}")

def demo_parrot_lightning_encoder_save_load():
    """Demo save/load for ParrotLightningEncoder (factory wrapper)."""
    print("\n" + "=" * 60)
    print("DEMO: ParrotLightningEncoder Save/Load (Factory wrapper)")
    print("=" * 60)
    
    # Create via the factory
    config = OmegaConf.create({
        "type": "table",
        "alphabet": "ARNDCQEGHILKMFPSTWYV"  # Standard 20 amino acids
    })
    
    original_encoder = ParrotLightningEncoder(config)
    print(f"Factory encoder type: {original_encoder.encoder_type}")
    print(f"Alphabet size: {len(original_encoder.get_alphabet())}")
    
    test_sequence = "ARNDCQ"
    encoded = original_encoder.encode(test_sequence)
    
    # Save and load via the factory methods
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "lightning_encoder.pkl"
        original_encoder.save(str(save_path))
        
        # Load using the factory's static method
        loaded_encoder = ParrotLightningEncoder.load(str(save_path))
        print(f"Loaded factory encoder type: {loaded_encoder.encoder_type}")
        
        # Test multi-sequence functionality
        sequences = ["ARND", "CQEG", "HIL"]
        original_batch = original_encoder.encode_sequences_padded(sequences)
        loaded_batch = loaded_encoder.encode_sequences_padded(sequences)
        
        print(f"Batch tensors identical: {torch.equal(original_batch, loaded_batch)}")

def demo_error_handling():
    """Demo error handling for save/load operations."""
    print("\n" + "=" * 60)
    print("DEMO: Error Handling")
    print("=" * 60)
    
    # Try to load a non-existent file
    try:
        BaseParrotEncoder.load("non_existent_file.pkl")
    except FileNotFoundError as e:
        print(f"Expected FileNotFoundError: {e}")
    
    # Try to save to an invalid location
    config = OmegaConf.create({"type": "table", "alphabet": "AC"})
    encoder = TableParrotEncoder(config)
    
    try:
        encoder.save("/invalid/path/encoder.pkl")
    except IOError as e:
        print(f"Expected IOError: {e}")

if __name__ == "__main__":
    print("PARROT Encoder Save/Load Functionality Demo")
    print("This demo shows how to save and load different types of encoders.")
    
    demo_table_encoder_save_load()
    demo_matrix_encoder_save_load()
    demo_parrot_lightning_encoder_save_load()
    demo_error_handling()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
