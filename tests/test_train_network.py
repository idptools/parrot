"""
Comprehensive tests for the train_network.py module.

This test suite covers:
1. Main training function with different configurations
2. Helper functions like get_collate_function
3. Different network types (BRNN vs UNet)
4. Different data types (sequence, residues, matrix)
5. Different problem types (regression, classification)
6. Cross-validation functionality
7. Error handling and edge cases

.............................................................................
parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import pytest
import tempfile
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as L
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from unittest.mock import Mock, patch, MagicMock

# Import the functions and classes we're testing
from parrot.train_network import (
    train,
    get_collate_function,
    matrix_collate,
    _train_lightning_unet,
    _train_traditional_brnn,
    _train_with_cross_validation,
    _train_brnn_with_cross_validation
)

from parrot.process_input_data import (
    SequenceDataset,
    seq_regress_collate,
    seq_class_collate,
    res_regress_collate,
    res_class_collate
)

from parrot.encode_sequence import ParrotLightningEncoder
from parrot.brnn_architecture import BRNN_PARROT, BRNN_PARROT_LEGACY
from parrot.unet_architecture import UNet_PARROT

# --- Test Data Definitions ---

# Sample sequence data for testing
SEQUENCE_REGRESSION_DATA = """# Test sequence regression data
seq1	ACDEFG	1.5
seq2	GHIKLM	2.3
seq3	NPQRST	0.8
seq4	VWYA	1.2
"""

SEQUENCE_CLASSIFICATION_DATA = """# Test sequence classification data
seq1	ACDEFG	0
seq2	GHIKLM	1
seq3	NPQRST	0
seq4	VWYA	1
"""

RESIDUE_REGRESSION_DATA = """# Test residue regression data
seq1	ACDE	1.1	2.2	3.3	4.4
seq2	FGHI	0.5	1.5	2.5	3.5
seq3	KLMN	2.0	1.0	3.0	4.0
"""

RESIDUE_CLASSIFICATION_DATA = """# Test residue classification data
seq1	ACDE	0	1	0	1
seq2	FGHI	1	0	1	0
seq3	KLMN	0	1	1	0
"""

# Matrix data - simplified for testing
MATRIX_REGRESSION_DATA = """# Test matrix regression data
seq1	ACDE	1.1	2.2	3.3	4.4	1.5	2.5	3.5	4.5	2.0	3.0	4.0	5.0	2.5	3.5	4.5	5.5	1.5
seq2	FGHI	0.5	1.5	2.5	3.5	1.0	2.0	3.0	4.0	1.5	2.5	3.5	4.5	2.0	3.0	4.0	5.0	2.3
"""

# --- Fixture Definitions ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def encoder_config():
    """Create a basic encoder configuration."""
    return OmegaConf.create({
        "type": "table",
        "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "table_type": "one_hot"
    })

@pytest.fixture
def matrix_encoder_config():
    """Create a matrix encoder configuration."""
    return OmegaConf.create({
        "type": "matrix",
        "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "encoding_type": "vectorial",
        "gap_char": "*",
        "use_gap": False
    })

@pytest.fixture
def mock_brnn_lightning():
    """Create a mock Lightning BRNN network."""
    mock_network = Mock(spec=BRNN_PARROT)
    mock_network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    mock_network.device = torch.device('cpu')
    return mock_network

@pytest.fixture
def mock_brnn_legacy():
    """Create a mock legacy BRNN network."""
    mock_network = Mock(spec=BRNN_PARROT_LEGACY)
    mock_network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    mock_network.device = torch.device('cpu')
    mock_network.float.return_value = mock_network
    return mock_network

@pytest.fixture
def mock_unet():
    """Create a mock UNet network."""
    mock_network = Mock(spec=UNet_PARROT)
    mock_network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    mock_network.device = torch.device('cpu')
    return mock_network

def create_test_dataset(data_content, temp_dir, encoder_config, datatype='sequence'):
    """Helper function to create test datasets."""
    # Create test data file
    data_file = os.path.join(temp_dir, 'test_data.tsv')
    with open(data_file, 'w') as f:
        f.write(data_content)
    
    # Create encoder
    encoder = ParrotLightningEncoder(encoder_config)
    
    # Create dataset
    dataset = SequenceDataset(
        filepath=data_file,
        encoder=encoder,
        datatype=datatype
    )
    
    return dataset

# --- Test Classes ---

class TestCollateFunction:
    """Test the get_collate_function helper."""
    
    def test_sequence_regression_collate(self):
        """Test that sequence regression returns correct collate function."""
        collate_fn = get_collate_function('sequence', 'regression')
        assert collate_fn == seq_regress_collate
    
    def test_sequence_classification_collate(self):
        """Test that sequence classification returns correct collate function."""
        collate_fn = get_collate_function('sequence', 'classification')
        assert collate_fn == seq_class_collate
    
    def test_residues_regression_collate(self):
        """Test that residue regression returns correct collate function."""
        collate_fn = get_collate_function('residues', 'regression')
        assert collate_fn == res_regress_collate
    
    def test_residues_classification_collate(self):
        """Test that residue classification returns correct collate function."""
        collate_fn = get_collate_function('residues', 'classification')
        assert collate_fn == res_class_collate
    
    def test_matrix_collate(self):
        """Test that matrix data returns matrix collate function."""
        collate_fn = get_collate_function('matrix', 'regression')
        assert collate_fn == matrix_collate
        
        collate_fn = get_collate_function('matrix', 'classification')
        assert collate_fn == matrix_collate
    
    def test_invalid_datatype(self):
        """Test that invalid datatype raises appropriate error."""
        with pytest.raises(ValueError):
            get_collate_function('invalid', 'regression')


class TestMatrixCollate:
    """Test the matrix_collate function."""
    
    def test_matrix_collate_regression(self):
        """Test matrix collate for regression tasks."""
        # Create mock batch data
        batch = [
            ('seq1', torch.randn(4, 4, 1), 1.5),
            ('seq2', torch.randn(4, 4, 1), 2.3)
        ]
        
        names, matrices, targets = matrix_collate(batch)
        
        assert len(names) == 2
        assert names == ['seq1', 'seq2']
        assert matrices.shape == (2, 4, 4, 1)
        assert targets.shape == (2,)
        assert torch.allclose(targets, torch.tensor([1.5, 2.3]))
    
    def test_matrix_collate_classification(self):
        """Test matrix collate for classification tasks."""
        # Create mock batch data with integer targets
        batch = [
            ('seq1', torch.randn(4, 4, 1), 0),
            ('seq2', torch.randn(4, 4, 1), 1)
        ]
        
        names, matrices, targets = matrix_collate(batch)
        
        assert len(names) == 2
        assert matrices.shape == (2, 4, 4, 1)
        assert targets.shape == (2,)
        assert torch.equal(targets, torch.tensor([0, 1]))


class TestMainTrainingFunction:
    """Test the main training function with different configurations."""
    
    @pytest.mark.parametrize("network_type", ["lightning_brnn", "unet", "legacy_brnn"])
    @pytest.mark.parametrize("datatype", ["sequence", "residues"])
    @pytest.mark.parametrize("problem_type", ["regression", "classification"])
    def test_training_different_configurations(self, temp_dir, encoder_config, 
                                             network_type, datatype, problem_type):
        """Test training with different network types, data types, and problem types."""
        
        # Choose appropriate test data
        if datatype == 'sequence' and problem_type == 'regression':
            data_content = SEQUENCE_REGRESSION_DATA
        elif datatype == 'sequence' and problem_type == 'classification':
            data_content = SEQUENCE_CLASSIFICATION_DATA
        elif datatype == 'residues' and problem_type == 'regression':
            data_content = RESIDUE_REGRESSION_DATA
        else:  # residues + classification
            data_content = RESIDUE_CLASSIFICATION_DATA
        
        # Create datasets
        train_dataset = create_test_dataset(data_content, temp_dir, encoder_config, datatype)
        val_dataset = create_test_dataset(data_content, temp_dir, encoder_config, datatype)
        
        # Create appropriate network mock
        if network_type == "lightning_brnn":
            network = Mock(spec=BRNN_PARROT)
        elif network_type == "unet":
            network = Mock(spec=UNet_PARROT)
        else:  # legacy_brnn
            network = Mock(spec=BRNN_PARROT_LEGACY)
            network.float.return_value = network
        
        network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        # Mock the actual training functions to avoid full training
        with patch('parrot.train_network._train_lightning_unet') as mock_lightning, \
             patch('parrot.train_network._train_traditional_brnn') as mock_traditional:
            
            mock_lightning.return_value = [0.1, 0.2, 0.15]
            mock_traditional.return_value = [0.1, 0.2, 0.15]
            
            result = train(
                network=network,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype=datatype,
                problem_type=problem_type,
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2,
                verbose=False,
                silent=True,
                batch_size=2
            )
            
            # Verify that the appropriate training function was called
            if network_type in ["lightning_brnn", "unet"]:
                mock_lightning.assert_called_once()
                mock_traditional.assert_not_called()
            else:
                mock_traditional.assert_called_once()
                mock_lightning.assert_not_called()
            
            # Check return value
            assert isinstance(result, list)
            assert len(result) == 3
    
    def test_training_with_matrix_data(self, temp_dir, matrix_encoder_config):
        """Test training with matrix data (UNet only)."""
        
        # Create matrix dataset
        train_dataset = create_test_dataset(
            MATRIX_REGRESSION_DATA, temp_dir, matrix_encoder_config, 'matrix'
        )
        val_dataset = create_test_dataset(
            MATRIX_REGRESSION_DATA, temp_dir, matrix_encoder_config, 'matrix'
        )
        
        # Create UNet mock
        network = Mock(spec=UNet_PARROT)
        network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with patch('parrot.train_network._train_lightning_unet') as mock_lightning:
            mock_lightning.return_value = [0.1, 0.2, 0.15]
            
            result = train(
                network=network,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='matrix',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2,
                batch_size=2
            )
            
            mock_lightning.assert_called_once()
            assert isinstance(result, list)
    
    @pytest.mark.parametrize("stop_condition", ["iter", "auto"])
    def test_different_stop_conditions(self, temp_dir, encoder_config, stop_condition):
        """Test training with different stop conditions."""
        
        train_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        network = Mock(spec=BRNN_PARROT_LEGACY)
        network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        network.float.return_value = network
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with patch('parrot.train_network._train_traditional_brnn') as mock_train:
            mock_train.return_value = [0.1, 0.2, 0.15]
            
            result = train(
                network=network,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='sequence',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition=stop_condition,
                device='cpu',
                learn_rate=0.001,
                n_epochs=5,
                batch_size=2
            )
            
            # Verify stop_condition was passed correctly
            call_args = mock_train.call_args
            assert call_args[1]['stop_condition'] == stop_condition
    
    def test_cross_validation_lightning(self, temp_dir, encoder_config):
        """Test cross-validation with Lightning networks."""
        
        train_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        network = Mock(spec=BRNN_PARROT)
        network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with patch('parrot.train_network._train_with_cross_validation') as mock_cv:
            mock_cv.return_value = {
                'fold_train_losses': [[0.1, 0.2], [0.15, 0.25]],
                'fold_val_losses': [[0.2, 0.3], [0.25, 0.35]],
                'fold_final_val_loss': [0.3, 0.35],
                'mean_val_loss': 0.325,
                'std_val_loss': 0.025
            }
            
            result = train(
                network=network,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='sequence',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2,
                cross_validation=True,
                cv_folds=2
            )
            
            mock_cv.assert_called_once()
            assert isinstance(result, dict)
            assert 'mean_val_loss' in result
    
    def test_cross_validation_legacy(self, temp_dir, encoder_config):
        """Test cross-validation with legacy BRNN networks."""
        
        train_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        network = Mock(spec=BRNN_PARROT_LEGACY)
        network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        network.float.return_value = network
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with patch('parrot.train_network._train_brnn_with_cross_validation') as mock_cv:
            mock_cv.return_value = {
                'fold_train_losses': [[0.1, 0.2], [0.15, 0.25]],
                'fold_val_losses': [[0.2, 0.3], [0.25, 0.35]],
                'fold_final_val_loss': [0.3, 0.35],
                'mean_val_loss': 0.325,
                'std_val_loss': 0.025
            }
            
            result = train(
                network=network,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='sequence',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2,
                cross_validation=True,
                cv_folds=2
            )
            
            mock_cv.assert_called_once()
            assert isinstance(result, dict)
    
    def test_with_dataloader_input(self, temp_dir, encoder_config):
        """Test training when datasets are already DataLoaders."""
        
        # Create datasets first
        train_ds = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_ds = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        # Convert to DataLoaders
        train_loader = DataLoader(train_ds, batch_size=2, collate_fn=seq_regress_collate)
        val_loader = DataLoader(val_ds, batch_size=2, collate_fn=seq_regress_collate)
        
        network = Mock(spec=BRNN_PARROT_LEGACY)
        network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        network.float.return_value = network
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with patch('parrot.train_network._train_traditional_brnn') as mock_train:
            mock_train.return_value = [0.1, 0.2, 0.15]
            
            result = train(
                network=network,
                train_dataset=train_loader,
                val_dataset=val_loader,
                datatype='sequence',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2
            )
            
            mock_train.assert_called_once()
            # Verify DataLoaders were passed through
            call_args = mock_train.call_args
            assert isinstance(call_args[0][1], DataLoader)  # train_dataset
            assert isinstance(call_args[0][2], DataLoader)  # val_dataset


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_datatype(self, temp_dir, encoder_config, mock_brnn_legacy):
        """Test that invalid datatype raises appropriate error."""
        
        train_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with pytest.raises(ValueError):
            train(
                network=mock_brnn_legacy,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='invalid_datatype',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2
            )
    
    def test_invalid_problem_type(self, temp_dir, encoder_config, mock_brnn_legacy):
        """Test that invalid problem type raises appropriate error."""
        
        train_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with pytest.raises(ValueError):
            train(
                network=mock_brnn_legacy,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='sequence',
                problem_type='invalid_problem_type',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2
            )
    
    def test_matrix_data_with_non_lightning_network(self, temp_dir, matrix_encoder_config):
        """Test that matrix data with non-Lightning network raises error."""
        
        train_dataset = create_test_dataset(
            MATRIX_REGRESSION_DATA, temp_dir, matrix_encoder_config, 'matrix'
        )
        val_dataset = create_test_dataset(
            MATRIX_REGRESSION_DATA, temp_dir, matrix_encoder_config, 'matrix'
        )
        
        # Use legacy BRNN (non-Lightning) with matrix data
        network = Mock(spec=BRNN_PARROT_LEGACY)
        network.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        network.float.return_value = network
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        with pytest.raises((ValueError, AttributeError)):
            train(
                network=network,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='matrix',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2
            )
    
    def test_empty_datasets(self, temp_dir, encoder_config, mock_brnn_legacy):
        """Test behavior with empty datasets."""
        
        # Create empty data file
        empty_data = "# Empty data file\n"
        empty_dataset = create_test_dataset(empty_data, temp_dir, encoder_config, 'sequence')
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        # This should either raise an error or handle gracefully
        # The exact behavior depends on implementation
        with pytest.raises((IndexError, ValueError, RuntimeError)):
            train(
                network=mock_brnn_legacy,
                train_dataset=empty_dataset,
                val_dataset=empty_dataset,
                datatype='sequence',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=2
            )


class TestIntegrationScenarios:
    """Integration tests that test realistic training scenarios."""
    
    def test_minimal_training_run(self, temp_dir, encoder_config):
        """Test a minimal but complete training run."""
        
        # Create small datasets
        train_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        # Create a simple mock network that mimics real behavior
        class SimpleMockNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(20, 1)  # 20 for one-hot encoding
            
            def forward(self, x):
                return self.linear(x.mean(dim=1))  # Simple aggregation
        
        network = SimpleMockNetwork()
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        # Mock the internal training loop but let most logic run
        with patch('torch.optim.Adam') as mock_optimizer_class, \
             patch('torch.save') as mock_save:
            
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            # This should run without errors
            result = train(
                network=network,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='sequence',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='iter',
                device='cpu',
                learn_rate=0.001,
                n_epochs=1,  # Just one epoch for speed
                verbose=False,
                silent=True,
                batch_size=2
            )
            
            # Verify that training completed and saved weights
            mock_save.assert_called()
            assert isinstance(result, list)
    
    def test_parameter_validation(self, temp_dir, encoder_config, mock_brnn_legacy):
        """Test that all parameters are properly validated and passed through."""
        
        train_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        val_dataset = create_test_dataset(
            SEQUENCE_REGRESSION_DATA, temp_dir, encoder_config, 'sequence'
        )
        
        weights_file = os.path.join(temp_dir, 'test_weights.pt')
        
        # Test with all parameters specified
        with patch('parrot.train_network._train_traditional_brnn') as mock_train:
            mock_train.return_value = [0.1, 0.2, 0.15]
            
            result = train(
                network=mock_brnn_legacy,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                datatype='sequence',
                problem_type='regression',
                weights_file=weights_file,
                stop_condition='auto',
                device='cuda',
                learn_rate=0.01,
                n_epochs=10,
                verbose=True,
                silent=False,
                batch_size=16,
                encoder_cfg=encoder_config,
                cross_validation=False,
                cv_folds=3
            )
            
            # Verify all parameters were passed correctly
            mock_train.assert_called_once()
            call_args = mock_train.call_args
            
            assert call_args[1]['stop_condition'] == 'auto'
            assert call_args[1]['device'] == 'cuda'
            assert call_args[1]['learn_rate'] == 0.01
            assert call_args[1]['n_epochs'] == 10
            assert call_args[1]['verbose'] == True
            assert call_args[1]['silent'] == False
            assert call_args[1]['batch_size'] == 16
            assert call_args[1]['cross_validation'] == False
            assert call_args[1]['cv_folds'] == 3


if __name__ == '__main__':
    pytest.main([__file__])
