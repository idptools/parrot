"""
Unit and regression tests for the UNet_PARROT architecture and its components.

This test suite covers:
- DoubleConv block functionality and parameter validation
- Down block functionality and parameter validation  
- Up block functionality and parameter validation
- UNet_PARROT model architecture and forward pass
- UNet_PARROT training, validation, and test steps
- Parameter validation for all classes
"""

import pytest
import torch
import torch.nn as nn
import pytorch_lightning as L
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import parrot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import specific classes directly from parrot
try:
    from parrot.unet_architecture import DoubleConv, Down, Up, UNet_PARROT
except ImportError:
    # If that fails, try importing directly from the file
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "unet_architecture", 
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parrot", "unet_architecture.py")
    )
    unet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unet_module)
    DoubleConv = unet_module.DoubleConv
    Down = unet_module.Down
    Up = unet_module.Up
    UNet_PARROT = unet_module.UNet_PARROT


class TestDoubleConv:
    """Test the DoubleConv block used in UNet encoder and decoder paths."""
    
    def test_doubleconv_basic_functionality(self):
        """Test basic forward pass through DoubleConv block."""
        # Create a simple DoubleConv block
        conv_block = DoubleConv(in_channels=3, out_channels=64)
        
        # Create sample input
        batch_size = 2
        height, width = 32, 32
        x = torch.randn(batch_size, 3, height, width)
        
        # Forward pass
        output = conv_block(x)
        
        # Check output shape
        assert output.shape == (batch_size, 64, height, width)
        
    def test_doubleconv_with_dropout(self):
        """Test DoubleConv block with dropout."""
        conv_block = DoubleConv(in_channels=3, out_channels=64, dropout=0.2)
        x = torch.randn(2, 3, 32, 32)
        
        output = conv_block(x)
        assert output.shape == (2, 64, 32, 32)
        
    def test_doubleconv_custom_kernel_size(self):
        """Test DoubleConv block with custom kernel size."""
        conv_block = DoubleConv(in_channels=3, out_channels=64, kernel_size=5)
        x = torch.randn(2, 3, 32, 32)
        
        output = conv_block(x)
        assert output.shape == (2, 64, 32, 32)
        
    def test_doubleconv_parameter_validation(self):
        """Test parameter validation for DoubleConv."""
        # Test invalid in_channels
        with pytest.raises(ValueError, match="in_channels must be a positive integer"):
            DoubleConv(in_channels=0, out_channels=64)
            
        with pytest.raises(ValueError, match="in_channels must be a positive integer"):
            DoubleConv(in_channels=-1, out_channels=64)
            
        with pytest.raises(ValueError, match="in_channels must be a positive integer"):
            DoubleConv(in_channels="invalid", out_channels=64)
            
        # Test invalid out_channels
        with pytest.raises(ValueError, match="out_channels must be a positive integer"):
            DoubleConv(in_channels=3, out_channels=0)
            
        # Test invalid dropout
        with pytest.raises(ValueError, match="dropout must be a non-negative float or None"):
            DoubleConv(in_channels=3, out_channels=64, dropout=-0.1)
            
        # Test invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
            DoubleConv(in_channels=3, out_channels=64, kernel_size=0)
            
        # Test even kernel_size
        with pytest.raises(ValueError, match="kernel_size must be an odd integer"):
            DoubleConv(in_channels=3, out_channels=64, kernel_size=4)


class TestDown:
    """Test the Down block used in UNet encoder path."""
    
    def test_down_basic_functionality(self):
        """Test basic forward pass through Down block."""
        down_block = Down(in_channels=64, out_channels=128)
        
        # Input will be downsampled by factor of 2
        x = torch.randn(2, 64, 32, 32)
        output = down_block(x)
        
        # Check output shape (spatial dimensions halved, channels changed)
        assert output.shape == (2, 128, 16, 16)
        
    def test_down_with_dropout_and_kernel_size(self):
        """Test Down block with dropout and custom kernel size."""
        down_block = Down(in_channels=64, out_channels=128, dropout=0.1, kernel_size=5)
        x = torch.randn(2, 64, 64, 64)
        
        output = down_block(x)
        assert output.shape == (2, 128, 32, 32)
        
    def test_down_parameter_validation(self):
        """Test parameter validation for Down block."""
        # Test invalid parameters (should propagate from DoubleConv)
        with pytest.raises(ValueError):
            Down(in_channels=0, out_channels=128)
            
        with pytest.raises(ValueError):
            Down(in_channels=64, out_channels=0)
            
        with pytest.raises(ValueError):
            Down(in_channels=64, out_channels=128, kernel_size=2)


class TestUp:
    """Test the Up block used in UNet decoder path."""
    
    def test_up_bilinear_functionality(self):
        """Test Up block with bilinear upsampling."""
        # With new Up logic: after upsampling in_channels -> in_channels//2
        # Skip connection has in_channels channels  
        # Total after concatenation: in_channels//2 + in_channels = 3*in_channels//2
        up_block = Up(in_channels=128, out_channels=64, bilinear=True)
        
        # x1 is from previous decoder layer, x2 is skip connection  
        x1 = torch.randn(2, 128, 16, 16)  # From decoder
        x2 = torch.randn(2, 128, 32, 32)  # Skip connection (should be in_channels)
        
        output = up_block(x1, x2)
        
        # Output should match x2 spatial dimensions
        assert output.shape == (2, 64, 32, 32)
        
    def test_up_transpose_conv_functionality(self):
        """Test Up block with transposed convolution."""
        up_block = Up(in_channels=128, out_channels=64, bilinear=False)
        
        x1 = torch.randn(2, 128, 16, 16)
        x2 = torch.randn(2, 128, 32, 32)  # Skip connection (should be in_channels)
        
        output = up_block(x1, x2)
        assert output.shape == (2, 64, 32, 32)
        
    def test_up_size_mismatch_handling(self):
        """Test Up block handling of size mismatches between x1 and x2."""
        up_block = Up(in_channels=128, out_channels=64, bilinear=True)
        
        # Slightly different sizes to test padding
        x1 = torch.randn(2, 128, 15, 15)  # Will be upsampled to 30x30
        x2 = torch.randn(2, 128, 32, 32)  # Skip connection (should be in_channels)
        
        output = up_block(x1, x2)
        # Should match x2 size due to padding
        assert output.shape == (2, 64, 32, 32)
        
    def test_up_parameter_validation(self):
        """Test parameter validation for Up block."""
        # Test invalid bilinear parameter
        with pytest.raises(ValueError, match="bilinear must be a boolean"):
            Up(in_channels=128, out_channels=64, bilinear="invalid")
            
        # Test other invalid parameters
        with pytest.raises(ValueError):
            Up(in_channels=0, out_channels=64)
            
        with pytest.raises(ValueError):
            Up(in_channels=128, out_channels=0)


class TestUNetPARROT:
    """Test the complete UNet_PARROT model."""
    
    def test_unet_basic_initialization(self):
        """Test basic UNet_PARROT initialization."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=2,
            problem_type='classification',
            batch_size=8
        )
        
        assert model.input_channels == 1
        assert model.num_classes == 2
        assert model.problem_type == 'classification'
        assert model.batch_size == 8
        assert model.base_channels == 64  # default
        assert model.kernel_size == 3     # default
        
    def test_unet_custom_parameters(self):
        """Test UNet_PARROT with custom parameters."""
        model = UNet_PARROT(
            input_channels=3,
            num_classes=1,
            problem_type='regression',
            batch_size=16,
            bilinear=False,
            base_channels=32,
            first_down_channels=96,
            kernel_size=5,
            dropout=0.2
        )
        
        assert model.input_channels == 3
        assert model.num_classes == 1
        assert model.problem_type == 'regression'
        assert model.base_channels == 32
        assert model.first_down_channels == 96
        assert model.kernel_size == 5
        assert model.dropout == 0.2
        
    def test_unet_forward_pass_classification(self):
        """Test forward pass for classification."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=3,
            problem_type='classification',
            batch_size=4
        )
        
        # Input must be divisible by 16
        x = torch.randn(4, 1, 64, 64)
        
        output = model(x)
        
        # Output should have same spatial dimensions as input
        assert output.shape == (4, 3, 64, 64)
        
    def test_unet_forward_pass_regression(self):
        """Test forward pass for regression."""
        model = UNet_PARROT(
            input_channels=3,
            num_classes=1,
            problem_type='regression',
            batch_size=2
        )
        
        x = torch.randn(2, 3, 128, 128)
        output = model(x)
        
        assert output.shape == (2, 1, 128, 128)
        
    def test_unet_different_input_sizes(self):
        """Test UNet with different valid input sizes."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=1
        )
        
        # Test various sizes divisible by 16 (skip 16 due to BatchNorm issues with 1x1 features)
        for size in [32, 48, 64, 96, 128, 256]:
            x = torch.randn(1, 1, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)
            
    def test_unet_training_step_classification(self):
        """Test training step for classification."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=3,
            problem_type='classification',
            batch_size=2
        )
        
        # Mock batch data
        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 3, (2, 64, 64))
        batch = (inputs, targets)
        
        loss = model.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
    def test_unet_training_step_regression(self):
        """Test training step for regression."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=2
        )
        
        inputs = torch.randn(2, 1, 32, 32)
        targets = torch.randn(2, 1, 32, 32)
        batch = (inputs, targets)
        
        loss = model.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
    def test_unet_validation_step(self):
        """Test validation step."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=2,
            problem_type='classification',
            batch_size=2
        )
        
        inputs = torch.randn(2, 1, 48, 48)
        targets = torch.randint(0, 2, (2, 48, 48))
        batch = (inputs, targets)
        
        loss = model.validation_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        
    def test_unet_test_step(self):
        """Test test step."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=1
        )
        
        inputs = torch.randn(1, 1, 32, 32)
        targets = torch.randn(1, 1, 32, 32)
        batch = (inputs, targets)
        
        loss = model.test_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        
    def test_unet_parameter_validation(self):
        """Test parameter validation for UNet_PARROT."""
        # Test invalid input_channels
        with pytest.raises(ValueError, match="input_channels must be a positive integer"):
            UNet_PARROT(
                input_channels=0,
                num_classes=1,
                problem_type='regression',
                batch_size=1
            )
            
        # Test invalid num_classes
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            UNet_PARROT(
                input_channels=1,
                num_classes=0,
                problem_type='regression',
                batch_size=1
            )
            
        # Test invalid problem_type
        with pytest.raises(ValueError, match="problem_type must be either 'regression' or 'classification'"):
            UNet_PARROT(
                input_channels=1,
                num_classes=1,
                problem_type='invalid',
                batch_size=1
            )
            
        # Test invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            UNet_PARROT(
                input_channels=1,
                num_classes=1,
                problem_type='regression',
                batch_size=0
            )
            
        # Test invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
            UNet_PARROT(
                input_channels=1,
                num_classes=1,
                problem_type='regression',
                batch_size=1,
                kernel_size=0
            )
            
        # Test even kernel_size
        with pytest.raises(ValueError, match="kernel_size must be an odd integer"):
            UNet_PARROT(
                input_channels=1,
                num_classes=1,
                problem_type='regression',
                batch_size=1,
                kernel_size=4
            )
            
        # Test invalid first_down_channels
        with pytest.raises(ValueError, match="first_down_channels must be a positive integer"):
            UNet_PARROT(
                input_channels=1,
                num_classes=1,
                problem_type='regression',
                batch_size=1,
                first_down_channels=0
            )
            
    def test_unet_optimizer_configuration_sgd(self):
        """Test optimizer configuration with SGD."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=1,
            optimizer_name='SGD',
            learn_rate=0.01,
            momentum=0.9
        )
        
        # Mock trainer to avoid requiring full training setup
        model.trainer = Mock()
        model.trainer.max_epochs = 10
        
        optimizers, schedulers = model.configure_optimizers()
        
        assert len(optimizers) == 1
        assert len(schedulers) == 1
        assert isinstance(optimizers[0], torch.optim.SGD)
        
    def test_unet_optimizer_configuration_adamw(self):
        """Test optimizer configuration with AdamW."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=1,
            optimizer_name='AdamW',
            learn_rate=0.001
        )
        
        model.trainer = Mock()
        model.trainer.max_epochs = 10
        
        optimizers, schedulers = model.configure_optimizers()
        
        assert len(optimizers) == 1
        assert len(schedulers) == 1
        assert isinstance(optimizers[0], torch.optim.AdamW)
        
    def test_unet_invalid_optimizer(self):
        """Test invalid optimizer configuration."""
        with pytest.raises(ValueError, match="Invalid optimizer name"):
            UNet_PARROT(
                input_channels=1,
                num_classes=1,
                problem_type='regression',
                batch_size=1,
                optimizer_name='invalid_optimizer'
            )
            
    def test_unet_architecture_consistency(self):
        """Test that the UNet architecture is internally consistent."""
        model = UNet_PARROT(
            input_channels=3,
            num_classes=5,
            problem_type='classification',
            batch_size=4,
            base_channels=32,
            first_down_channels=48
        )
        
        # Check that the encoder-decoder channel progression is correct
        assert model.inc.double_conv[0].in_channels == 3
        assert model.inc.double_conv[0].out_channels == 32
        
        assert model.down1.maxpool_conv[1].double_conv[0].in_channels == 32
        assert model.down1.maxpool_conv[1].double_conv[0].out_channels == 48
        
        assert model.down2.maxpool_conv[1].double_conv[0].in_channels == 48
        assert model.down2.maxpool_conv[1].double_conv[0].out_channels == 96
        
        assert model.outc.in_channels == 32
        assert model.outc.out_channels == 5
        
    def test_unet_channel_progression_custom(self):
        """Test custom channel progression."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=1,
            base_channels=16,
            first_down_channels=32
        )
        
        # Test forward pass to ensure architecture works
        x = torch.randn(1, 1, 64, 64)
        output = model(x)
        assert output.shape == (1, 1, 64, 64)
        
    def test_unet_default_first_down_channels(self):
        """Test that first_down_channels defaults to base_channels * 2."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=1,
            problem_type='regression',
            batch_size=1,
            base_channels=32
            # first_down_channels not specified
        )
        
        assert model.first_down_channels == 64  # base_channels * 2


class TestUNetIntegration:
    """Integration tests for the complete UNet system."""
    
    def test_unet_end_to_end_classification(self):
        """Test end-to-end functionality for classification."""
        model = UNet_PARROT(
            input_channels=1,
            num_classes=3,
            problem_type='classification',
            batch_size=2,
            base_channels=16  # Smaller for faster testing
        )
        
        # Test forward pass
        x = torch.randn(2, 1, 32, 32)
        output = model(x)
        assert output.shape == (2, 3, 32, 32)
        
        # Test training step
        targets = torch.randint(0, 3, (2, 32, 32))
        batch = (x, targets)
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        
        # Test validation step
        val_loss = model.validation_step(batch, 0)
        assert isinstance(val_loss, torch.Tensor)
        
    def test_unet_end_to_end_regression(self):
        """Test end-to-end functionality for regression."""
        model = UNet_PARROT(
            input_channels=3,
            num_classes=1,
            problem_type='regression',
            batch_size=1,
            base_channels=8,
            first_down_channels=16
        )
        
        # Test forward pass
        x = torch.randn(1, 3, 48, 48)
        output = model(x)
        assert output.shape == (1, 1, 48, 48)
        
        # Test training step
        targets = torch.randn(1, 1, 48, 48)
        batch = (x, targets)
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        
        # Test that gradients flow properly
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients found - gradient flow may be broken"


if __name__ == "__main__":
    pytest.main([__file__])
