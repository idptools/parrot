#!/usr/bin/env python3

import torch
import sys
import os

# Add the parrot package to the path so we can import the UNet classes
sys.path.insert(0, '/home/nrazo/packages/parrot')

from parrot.unet_architecture import UNet_PARROT

def test_different_configurations():
    """Test UNet with different configurations to ensure robustness."""
    
    print("=== Testing Different UNet Configurations ===")
    
    configurations = [
        # (input_channels, num_classes, problem_type, input_size, base_channels, first_down_channels)
        (1, 1, 'regression', 64, 64, 128),  # Default
        (3, 3, 'classification', 128, 32, 64),  # RGB input, 3 classes
        (1, 5, 'classification', 96, 16, 32),  # 5 classes, smaller channels
        (4, 1, 'regression', 256, 8, 16),  # 4 channels input, very small channels
    ]
    
    for i, (input_channels, num_classes, problem_type, input_size, base_channels, first_down_channels) in enumerate(configurations):
        print(f"\n--- Configuration {i+1} ---")
        print(f"Input: {input_channels} channels, {input_size}x{input_size}")
        print(f"Output: {num_classes} classes, {problem_type}")
        print(f"Architecture: base_channels={base_channels}, first_down_channels={first_down_channels}")
        
        try:
            # Create model
            model = UNet_PARROT(
                input_channels=input_channels,
                num_classes=num_classes,
                problem_type=problem_type,
                batch_size=2,
                base_channels=base_channels,
                first_down_channels=first_down_channels
            )
            
            # Test forward pass
            x = torch.randn(2, input_channels, input_size, input_size)
            output = model(x)
            
            expected_shape = (2, num_classes, input_size, input_size)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            print(f"✓ SUCCESS: {x.shape} -> {output.shape}")
            
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False
    
    print(f"\n🎉 All {len(configurations)} configurations passed!")
    return True

if __name__ == "__main__":
    success = test_different_configurations()
    if not success:
        sys.exit(1)
