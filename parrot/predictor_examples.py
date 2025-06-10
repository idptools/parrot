"""
Example usage of the new PARROT base predictor classes.

This file demonstrates how to use the new base predictor architecture for
creating custom predictors and using them with different types of models.
"""

import numpy as np
from pathlib import Path

# Import the new base predictor classes
from parrot.base_predictor import BasePredictor, LegacyBRNNPredictor, LightningPredictor, create_predictor


def example_legacy_predictor():
    """Example of using the legacy predictor for backward compatibility."""
    print("=== Legacy Predictor Example ===")
    
    # This works with old-style PARROT model files
    # model_path = "path/to/your/legacy_model.pt"
    # predictor = LegacyBRNNPredictor(model_path)
    # predictor.datatype = 'sequence'  # Must set this for legacy models
    # predictor._initialize_network()
    
    # Make predictions
    # sequence = "ACDEFGHIKLMNPQRSTVWY"
    # prediction = predictor.predict(sequence)
    # print(f"Prediction for {sequence}: {prediction}")
    
    print("Legacy predictor example (commented out - requires actual model file)")


def example_lightning_predictor():
    """Example of using the Lightning predictor for new models."""
    print("\n=== Lightning Predictor Example ===")
    
    # This works with PyTorch Lightning checkpoint files
    # model_path = "path/to/your/lightning_model.ckpt"
    # predictor = LightningPredictor(model_path)
    
    # Make predictions
    # sequence = "ACDEFGHIKLMNPQRSTVWY"
    # prediction = predictor.predict(sequence)
    # print(f"Prediction for {sequence}: {prediction}")
    
    print("Lightning predictor example (commented out - requires actual model file)")


def example_factory_function():
    """Example of using the factory function to auto-detect model type."""
    print("\n=== Factory Function Example ===")
    
    # The factory function automatically detects the model type
    # model_path = "path/to/your/model.pt"  # or .ckpt
    # predictor = create_predictor(model_path, datatype='sequence')  # datatype only needed for legacy models
    
    # Make predictions
    # sequence = "ACDEFGHIKLMNPQRSTVWY"
    # prediction = predictor.predict(sequence)
    # print(f"Prediction for {sequence}: {prediction}")
    
    print("Factory function example (commented out - requires actual model file)")


def example_batch_predictions():
    """Example of making batch predictions."""
    print("\n=== Batch Predictions Example ===")
    
    # Load predictor (commented out - requires actual model)
    # predictor = create_predictor("path/to/model.pt", datatype='sequence')
    
    # Prepare multiple sequences
    # sequences = [
    #     "ACDEFGHIKLMNPQRSTVWY",
    #     "GGGGGGGGGGGGGGGGGGGG",
    #     "PPPPPPPPPPPPPPPPPPPP"
    # ]
    
    # Make batch predictions
    # predictions = predictor.predict_batch(sequences)
    # for seq, pred in zip(sequences, predictions):
    #     print(f"Sequence: {seq[:20]}... -> Prediction: {pred}")
    
    print("Batch predictions example (commented out - requires actual model file)")


def example_file_predictions():
    """Example of making predictions from a file."""
    print("\n=== File Predictions Example ===")
    
    # Load predictor (commented out - requires actual model)
    # predictor = create_predictor("path/to/model.pt", datatype='sequence')
    
    # Make predictions from file
    # input_file = "sequences.txt"  # Format: "seq_id\tsequence" per line
    # output_file = "predictions.txt"
    # predictions = predictor.predict_from_file(input_file, output_file)
    
    # Print some results
    # for seq_id, pred in list(predictions.items())[:5]:
    #     print(f"{seq_id}: {pred}")
    
    print("File predictions example (commented out - requires actual model and input files)")


class CustomPredictor(BasePredictor):
    """
    Example of creating a custom predictor by inheriting from BasePredictor.
    
    This shows how to extend the base class for specialized use cases.
    """
    
    def _extract_hyperparameters(self):
        """Custom hyperparameter extraction."""
        # Implement your custom logic here
        # For example, if you have a different checkpoint format
        self.hyperparameters = {
            'custom_param': 'custom_value',
            # ... extract your specific parameters
        }
        
        # Set required attributes
        self.datatype = 'sequence'  # or extract from checkpoint
        self.problem_type = 'regression'  # or extract from checkpoint
        self.num_classes = 1  # or extract from checkpoint
        
        # Setup encoding
        self._setup_encoding('onehot')
    
    def _initialize_network(self):
        """Custom network initialization."""
        # Create your custom network architecture
        # This is where you would instantiate your specific model
        # For now, we'll just create a placeholder
        pass  # Replace with actual network initialization
    
    def _postprocess_prediction(self, raw_output, sequence):
        """Custom postprocessing."""
        # Apply any custom postprocessing to the raw model output
        # For example, custom normalization, scaling, etc.
        prediction = raw_output.detach().cpu().numpy()
        
        # Apply custom transformations
        processed_prediction = self._apply_custom_transforms(prediction)
        
        return processed_prediction
    
    def _apply_custom_transforms(self, prediction):
        """Apply custom transformations to predictions."""
        # Example: apply some custom scaling or transformation
        return prediction * 2.0  # Just an example


def example_custom_predictor():
    """Example of using a custom predictor."""
    print("\n=== Custom Predictor Example ===")
    
    # Note: This is just a structural example
    # In practice, you would implement the abstract methods properly
    print("Custom predictor example - see CustomPredictor class for implementation details")


def example_model_info():
    """Example of getting model information."""
    print("\n=== Model Information Example ===")
    
    # Load predictor (commented out - requires actual model)
    # predictor = create_predictor("path/to/model.pt", datatype='sequence')
    
    # Get model information
    # info = predictor.get_model_info()
    # print("Model Information:")
    # for key, value in info.items():
    #     print(f"  {key}: {value}")
    
    print("Model information example (commented out - requires actual model file)")


def example_backward_compatibility():
    """Example showing backward compatibility with existing code."""
    print("\n=== Backward Compatibility Example ===")
    
    # The old interface still works:
    # from parrot import py_predictor
    # predictor = py_predictor.Predictor('/path/to/model.pt', dtype='sequence')
    # prediction = predictor.predict('ACDEFGHIKLMNPQRSTVWY')
    
    # But now it's powered by the new base class architecture!
    print("Backward compatibility maintained - old code should still work")


if __name__ == "__main__":
    print("PARROT Base Predictor Examples")
    print("=" * 50)
    
    # Run all examples
    example_legacy_predictor()
    example_lightning_predictor()
    example_factory_function()
    example_batch_predictions()
    example_file_predictions()
    example_custom_predictor()
    example_model_info()
    example_backward_compatibility()
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    print("\nTo use these examples with real models:")
    print("1. Replace the commented-out model paths with actual model files")
    print("2. Ensure the model files exist and are accessible")
    print("3. For legacy models, specify the correct datatype ('sequence' or 'residues')")
    print("4. For Lightning models, the datatype is automatically detected")
    
    print("\nKey benefits of the new base predictor architecture:")
    print("- Unified interface for different model types")
    print("- Automatic model type detection")
    print("- Extensible design for custom predictors")
    print("- Batch processing capabilities")
    print("- File I/O support")
    print("- Backward compatibility with existing code")
