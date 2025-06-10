"""
Base predictor class for PARROT neural networks

This module provides a base class for creating predictors that can work with any
PARROT network architecture. The base class handles common functionality like
model loading, validation, and prediction interfaces, while allowing specific
implementations to customize network architecture, preprocessing, and postprocessing.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""


import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path

from parrot import brnn_architecture
from parrot import encode_sequence
from parrot.tools import validate_args


class BasePredictor(ABC):
    """
    Abstract base class for PARROT network predictors.
    
    This class provides a common interface and shared functionality for all PARROT
    predictors. Specific predictor implementations should inherit from this class
    and implement the abstract methods to customize behavior for different
    network architectures and use cases.
    
    Attributes
    ----------
    model_path : Path
        Path to the saved model checkpoint
    device : str
        Device to run predictions on ('cpu' or 'cuda')
    datatype : str
        Type of data the model was trained on ('sequence' or 'residues')
    problem_type : str
        Type of problem ('regression' or 'classification')
    num_classes : int
        Number of output classes (1 for regression, >1 for classification)
    network : torch.nn.Module
        The loaded PyTorch network
    hyperparameters : dict
        Dictionary containing model hyperparameters
    encoding_scheme : str
        Encoding scheme used by the model
    encoder : object
        Encoder object for sequence encoding
    input_size : int
        Size of input vectors
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto',
        force_cpu: bool = False,
        **kwargs
    ):
        """
        Initialize the base predictor.
        
        Parameters
        ----------
        model_path : str or Path
            Path to the saved model checkpoint
        device : str, optional
            Device to use for predictions ('auto', 'cpu', 'cuda'), by default 'auto'
        force_cpu : bool, optional
            Force CPU usage even if GPU is available, by default False
        **kwargs
            Additional keyword arguments passed to specific implementations
        """
        self.model_path = Path(model_path)
        self.force_cpu = force_cpu
        self._validate_model_path()
        
        # Set device
        self.device = self._determine_device(device)
        
        # Initialize placeholders
        self.network = None
        self.hyperparameters = {}
        self.datatype = None
        self.problem_type = None
        self.num_classes = None
        self.encoding_scheme = None
        self.encoder = None
        self.input_size = None
        
        # Load model and extract metadata
        self._load_model()
        self._extract_hyperparameters()
        self._initialize_network()
        
        # Extract batch size for efficient batch processing
        self._extract_batch_size()
        
    def _validate_model_path(self) -> None:
        """Validate that the model path exists and is accessible."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not self.model_path.is_file():
            raise ValueError(f"Model path is not a file: {self.model_path}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the appropriate device for predictions."""
        if self.force_cpu:
            return 'cpu'
        
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device in ['cpu', 'cuda']:
            if device == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                return 'cpu'
            return device
        else:
            raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
    
    def _load_model(self) -> None:
        """Load the model checkpoint from disk."""
        try:
            self.checkpoint = torch.load(
                self.model_path, 
                map_location=torch.device(self.device),
                weights_only=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    @abstractmethod
    def _extract_hyperparameters(self) -> None:
        """
        Extract hyperparameters from the loaded model.
        
        This method should be implemented by subclasses to extract model-specific
        hyperparameters from the checkpoint. The implementation should populate
        self.hyperparameters with relevant information.
        """
        pass
    
    @abstractmethod
    def _initialize_network(self) -> None:
        """
        Initialize the network architecture.
        
        This method should be implemented by subclasses to create and configure
        the appropriate network architecture based on the extracted hyperparameters.
        The implementation should set self.network and load the model weights.
        """
        pass
    
    def _setup_encoding(self, encoding_scheme: str = 'onehot') -> None:
        """
        Set up sequence encoding scheme.
        
        Parameters
        ----------
        encoding_scheme : str, optional
            Encoding scheme to use, by default 'onehot'
        """
        self.encoding_scheme, self.encoder, self.input_size = (
            validate_args.set_encoding_scheme(encoding_scheme)
        )
    
    def _extract_batch_size(self) -> None:
        """
        Extract the training batch size from hyperparameters for efficient batch processing.
        
        For Lightning models, this is stored in hyperparameters. For legacy models,
        we use a reasonable default or try to infer from the model structure.
        """
        if hasattr(self, 'hyperparameters'):
            # Try to get batch size from hyperparameters (Lightning models)
            self.training_batch_size = self.hyperparameters.get('batch_size', None)
            
            # If not found, try common alternative names
            if self.training_batch_size is None:
                self.training_batch_size = self.hyperparameters.get('train_batch_size', None)
            if self.training_batch_size is None:
                self.training_batch_size = self.hyperparameters.get('dataloader_batch_size', None)
        else:
            self.training_batch_size = None
        
        # Set a reasonable default if no batch size found
        if self.training_batch_size is None:
            self.training_batch_size = 32  # Common default
            
        # Ensure it's a reasonable value for inference
        self.inference_batch_size = min(self.training_batch_size, 64)  # Cap at 64 for memory efficiency
    
    def _validate_sequence(self, sequence: str) -> str:
        """
        Validate and preprocess input sequence.
        
        Parameters
        ----------
        sequence : str
            Input amino acid sequence
            
        Returns
        -------
        str
            Validated and preprocessed sequence
            
        Raises
        ------
        ValueError
            If sequence contains invalid characters
        """
        if not isinstance(sequence, str):
            raise TypeError("Sequence must be a string")
        
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Validate amino acid characters
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(sequence) - valid_aas
        if invalid_chars:
            raise ValueError(f"Invalid amino acid characters found: {invalid_chars}")
        
        if len(sequence) == 0:
            raise ValueError("Sequence cannot be empty")
        
        return sequence
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode a sequence into a tensor suitable for the model.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence to encode
            
        Returns
        -------
        torch.Tensor
            Encoded sequence tensor
        """
        if self.encoding_scheme == 'onehot':
            encoded = encode_sequence.one_hot(sequence)
            # Reshape for batch processing: (1, sequence_length, input_size)
            encoded = encoded.view(1, len(sequence), -1)
        else:
            # Add support for other encoding schemes as needed
            raise NotImplementedError(f"Encoding scheme '{self.encoding_scheme}' not implemented")
        
        return encoded.to(self.device)
    
    def _encode_sequences_batch(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode multiple sequences into a batched tensor.
        
        This method handles variable-length sequences by padding them to the same length.
        
        Parameters
        ----------
        sequences : List[str]
            List of amino acid sequences to encode
            
        Returns
        -------
        torch.Tensor
            Batched encoded sequences tensor with shape (batch_size, max_seq_len, input_size)
        """
        if not sequences:
            raise ValueError("Empty sequence list provided")
        
        # Find the maximum sequence length for padding
        max_len = max(len(seq) for seq in sequences)
        
        if self.encoding_scheme == 'onehot':
            # Encode each sequence
            encoded_seqs = []
            for seq in sequences:
                encoded = encode_sequence.one_hot(seq)
                # Pad sequence if needed
                if len(seq) < max_len:
                    padding_size = max_len - len(seq)
                    padding = torch.zeros(padding_size, encoded.shape[1])
                    encoded = torch.cat([encoded, padding], dim=0)
                encoded_seqs.append(encoded)
            
            # Stack into batch
            batch_tensor = torch.stack(encoded_seqs, dim=0)
        else:
            raise NotImplementedError(f"Batch encoding for '{self.encoding_scheme}' not implemented")
        
        return batch_tensor.to(self.device)
    
    @abstractmethod
    def _postprocess_prediction(self, raw_output: torch.Tensor, sequence: str) -> np.ndarray:
        """
        Postprocess raw model output into final predictions.
        
        Parameters
        ----------
        raw_output : torch.Tensor
            Raw output from the neural network
        sequence : str
            Original input sequence
            
        Returns
        -------
        np.ndarray
            Processed predictions
        """
        pass
    
    def predict(self, sequence: str) -> np.ndarray:
        """
        Make predictions for a single sequence.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence to make predictions for
            
        Returns
        -------
        np.ndarray
            Predictions for the sequence
        """
        # Validate input
        sequence = self._validate_sequence(sequence)
        
        # Encode sequence
        encoded_seq = self._encode_sequence(sequence)
        
        # Set model to evaluation mode
        self.network.eval()
        
        # Make prediction
        with torch.no_grad():
            raw_output = self.network(encoded_seq.float())
        
        # Postprocess and return
        return self._postprocess_prediction(raw_output, sequence)
    
    def predict_batch_efficient(self, sequences: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Make predictions for a batch of sequences using efficient batching.
        
        This method processes sequences in batches using the detected training batch size
        or a specified batch size, which is much more efficient than processing them
        one by one, especially on GPU.
        
        Parameters
        ----------
        sequences : List[str]
            List of amino acid sequences
        batch_size : int, optional
            Batch size to use for processing. If None, uses the detected training batch size.
            
        Returns
        -------
        List[np.ndarray]
            List of predictions, one for each input sequence
        """
        if not sequences:
            return []
        
        # Use detected batch size or provided batch size
        if batch_size is None:
            batch_size = self.inference_batch_size
        
        # Validate all sequences first
        validated_sequences = [self._validate_sequence(seq) for seq in sequences]
        
        # Set model to evaluation mode
        self.network.eval()
        
        all_predictions = []
        
        # Process sequences in batches
        for i in range(0, len(validated_sequences), batch_size):
            batch_sequences = validated_sequences[i:i + batch_size]
            
            # Encode the batch
            batch_encoded = self._encode_sequences_batch(batch_sequences)
            
            # Make predictions for the batch
            with torch.no_grad():
                batch_output = self.network(batch_encoded.float())
            
            # Postprocess each sequence in the batch
            for j, seq in enumerate(batch_sequences):
                # Extract output for this sequence
                if self.datatype == 'sequence':
                    # For sequence prediction, take the j-th output
                    seq_output = batch_output[j:j+1]  # Keep batch dimension
                else:
                    # For residue prediction, take sequence length into account
                    seq_len = len(seq)
                    seq_output = batch_output[j:j+1, :seq_len]  # Remove padding
                
                prediction = self._postprocess_prediction(seq_output, seq)
                all_predictions.append(prediction)
        
        return all_predictions
    
    def predict_batch(self, sequences: List[str], use_efficient_batching: bool = True, batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Make predictions for a batch of sequences.
        
        Parameters
        ----------
        sequences : List[str]
            List of amino acid sequences
        use_efficient_batching : bool, optional
            Whether to use efficient batching (recommended), by default True
        batch_size : int, optional
            Batch size to use for efficient batching. If None, uses detected training batch size.
            
        Returns
        -------
        List[np.ndarray]
            List of predictions, one for each input sequence
        """
        if use_efficient_batching:
            return self.predict_batch_efficient(sequences, batch_size)
        else:
            # Fallback to original method (process one by one)
            predictions = []
            for sequence in sequences:
                predictions.append(self.predict(sequence))
            return predictions
    
    def predict_from_file(
        self, 
        input_file: Union[str, Path], 
        output_file: Optional[Union[str, Path]] = None,
        exclude_seq_id: bool = False,
        use_efficient_batching: bool = True,
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for sequences in a file.
        
        Parameters
        ----------
        input_file : str or Path
            Path to file containing sequences
        output_file : str or Path, optional
            Path to save predictions, by default None
        exclude_seq_id : bool, optional
            Whether sequence IDs are excluded from input file, by default False
        use_efficient_batching : bool, optional
            Whether to use efficient batching (recommended), by default True
        batch_size : int, optional
            Batch size to use for efficient batching. If None, uses detected training batch size.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping sequence identifiers to predictions
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Parse sequences from file
        sequences = {}
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                if exclude_seq_id:
                    seq_id = f"seq_{line_num}"
                    sequence = line
                else:
                    parts = line.split(None, 1)  # Split on whitespace, max 2 parts
                    if len(parts) != 2:
                        raise ValueError(f"Line {line_num}: Expected format '<id> <sequence>'")
                    seq_id, sequence = parts
                
                sequences[seq_id] = sequence
        
        # Extract sequences and IDs for batch processing
        seq_ids = list(sequences.keys())
        seq_list = list(sequences.values())
        
        # Make predictions using batch processing
        batch_predictions = self.predict_batch(seq_list, use_efficient_batching, batch_size)
        
        # Map predictions back to sequence IDs
        predictions = {}
        for seq_id, pred in zip(seq_ids, batch_predictions):
            predictions[seq_id] = pred
        
        # Save to file if requested
        if output_file is not None:
            self._save_predictions_to_file(predictions, output_file)
        
        return predictions
    
    def _save_predictions_to_file(
        self, 
        predictions: Dict[str, np.ndarray], 
        output_file: Union[str, Path]
    ) -> None:
        """
        Save predictions to a file.
        
        Parameters
        ----------
        predictions : Dict[str, np.ndarray]
            Dictionary of predictions
        output_file : str or Path
            Output file path
        """
        output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            for seq_id, pred in predictions.items():
                if self.datatype == 'sequence':
                    # For sequence prediction, write single value per sequence
                    if pred.ndim == 0:
                        f.write(f"{seq_id}\t{pred}\n")
                    else:
                        pred_str = '\t'.join(map(str, pred.flatten()))
                        f.write(f"{seq_id}\t{pred_str}\n")
                else:
                    # For residue prediction, write values for each position
                    pred_str = '\t'.join(map(str, pred.flatten()))
                    f.write(f"{seq_id}\t{pred_str}\n")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model information
        """
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'datatype': self.datatype,
            'problem_type': self.problem_type,
            'num_classes': self.num_classes,
            'encoding_scheme': self.encoding_scheme,
            'input_size': self.input_size,
            'training_batch_size': getattr(self, 'training_batch_size', None),
            'inference_batch_size': getattr(self, 'inference_batch_size', None),
            'hyperparameters': self.hyperparameters.copy()
        }
    
    def get_batch_info(self) -> Dict[str, Any]:
        """
        Get information about batch processing capabilities.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing batch processing information
        """
        return {
            'training_batch_size': getattr(self, 'training_batch_size', None),
            'inference_batch_size': getattr(self, 'inference_batch_size', None),
            'supports_efficient_batching': True,
            'recommended_batch_size': getattr(self, 'inference_batch_size', 32)
        }
    
    def __str__(self) -> str:
        """String representation of the predictor."""
        return (f"{self.__class__.__name__}("
                f"model_path='{self.model_path}', "
                f"datatype='{self.datatype}', "
                f"problem_type='{self.problem_type}', "
                f"device='{self.device}')")
    
    def __repr__(self) -> str:
        """Detailed string representation of the predictor."""
        return self.__str__()


class LegacyBRNNPredictor(BasePredictor):
    """
    Predictor for legacy BRNN models (compatible with existing py_predictor.Predictor).
    
    This class provides backward compatibility with the existing PARROT predictor
    interface while using the new base class architecture.
    """
    
    def _extract_hyperparameters(self) -> None:
        """Extract hyperparameters from legacy BRNN checkpoint."""
        # For legacy models, extract parameters from the state dict
        state_dict = self.checkpoint
        
        # Determine number of LSTM layers
        num_layers = 0
        while f'lstm.weight_ih_l{num_layers}' in state_dict:
            num_layers += 1
        
        # Extract hidden size (weight_ih has shape [4*hidden_size, input_size])
        lstm_weight_shape = state_dict['lstm.weight_ih_l0'].shape
        hidden_size = lstm_weight_shape[0] // 4
        
        # Extract number of classes from final layer
        if 'fc.bias' in state_dict:
            num_classes = state_dict['fc.bias'].shape[0]
        else:
            # For newer models with multiple linear layers, find the last linear layer
            linear_layers = [k for k in state_dict.keys() if 'linear_layers' in k and 'bias' in k]
            if linear_layers:
                # Get the highest numbered linear layer
                last_layer = max(linear_layers, key=lambda x: int(x.split('.')[1]))
                num_classes = state_dict[last_layer].shape[0]
            else:
                raise ValueError("Cannot determine number of classes from model")
        
        self.hyperparameters = {
            'num_lstm_layers': num_layers,
            'lstm_hidden_size': hidden_size,
            'num_classes': num_classes,
            'input_size': 20  # Assume one-hot encoding for legacy models
        }
        
        # Set basic attributes
        self.num_classes = num_classes
        self.problem_type = 'regression' if num_classes == 1 else 'classification'
        
        # Setup encoding (legacy models use one-hot)
        self._setup_encoding('onehot')
    
    def _initialize_network(self) -> None:
        """Initialize the legacy BRNN network."""
        # Extract hyperparameters
        num_layers = self.hyperparameters['num_lstm_layers']
        hidden_size = self.hyperparameters['lstm_hidden_size']
        num_classes = self.hyperparameters['num_classes']
        input_size = self.hyperparameters['input_size']
        
        # For legacy models, we need to determine datatype from usage context
        # Since this info isn't stored in old checkpoints, we'll need it passed in
        if not hasattr(self, 'datatype') or self.datatype is None:
            # Default to sequence for backward compatibility
            self.datatype = 'sequence'
        
        # Try to create legacy-style network first (for very old models)
        try:
            # Check if we're dealing with a very old model format
            # Old models used different constructor signatures
            if self.datatype == 'sequence':
                # Try old-style constructor first
                from parrot import brnn_architecture
                # Note: This might fail for new Lightning models
                self.network = brnn_architecture.BRNN_MtO(
                    input_size, hidden_size, num_layers, num_classes, self.device
                )
            elif self.datatype == 'residues':
                from parrot import brnn_architecture
                self.network = brnn_architecture.BRNN_MtM(
                    input_size, hidden_size, num_layers, num_classes, self.device
                )
            else:
                raise ValueError(f"Invalid datatype: {self.datatype}")
                
        except TypeError:
            # New model format - use Lightning-style constructor
            if self.datatype == 'sequence':
                self.network = brnn_architecture.BRNN_MtO(
                    input_size=input_size,
                    lstm_hidden_size=hidden_size,
                    num_lstm_layers=num_layers,
                    num_classes=num_classes,
                    problem_type=self.problem_type,
                    datatype=self.datatype,
                    batch_size=1  # For prediction, batch size is typically 1
                )
            elif self.datatype == 'residues':
                self.network = brnn_architecture.BRNN_MtM(
                    input_size=input_size,
                    lstm_hidden_size=hidden_size,
                    num_lstm_layers=num_layers,
                    num_classes=num_classes,
                    problem_type=self.problem_type,
                    datatype=self.datatype
                )
            else:
                raise ValueError(f"Invalid datatype: {self.datatype}")
        
        # Load the state dict
        self.network.load_state_dict(self.checkpoint)
        self.network.to(self.device)
        self.network.eval()
    
    def _postprocess_prediction(self, raw_output: torch.Tensor, sequence: str) -> np.ndarray:
        """Postprocess predictions for legacy BRNN models."""
        prediction = raw_output.detach().cpu().numpy()
        
        if self.problem_type == 'classification':
            if self.datatype == 'residues':
                # For residue classification, apply softmax to each position
                prediction = prediction.reshape(-1, self.num_classes)
                prediction = np.array([self._softmax(row) for row in prediction])
            else:
                # For sequence classification, apply softmax to the output
                prediction = self._softmax(prediction.flatten())
        else:
            # For regression, just flatten the output
            prediction = prediction.flatten()
        
        return prediction
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax function to array."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)


class LightningPredictor(BasePredictor):
    """
    Predictor for PyTorch Lightning BRNN models.
    
    This class handles models saved as PyTorch Lightning checkpoints with
    full hyperparameter information and modern architecture.
    """
    
    def _extract_hyperparameters(self) -> None:
        """Extract hyperparameters from Lightning checkpoint."""
        # Lightning checkpoints contain hyperparameters
        if 'hyper_parameters' in self.checkpoint:
            self.hyperparameters = self.checkpoint['hyper_parameters']
        else:
            raise ValueError("No hyperparameters found in Lightning checkpoint")
        
        # Extract key attributes
        self.datatype = self.hyperparameters.get('datatype')
        self.problem_type = self.hyperparameters.get('problem_type')
        self.num_classes = self.hyperparameters.get('num_classes')
        
        # Setup encoding
        encoding_scheme = self.hyperparameters.get('encoding_scheme', 'onehot')
        self._setup_encoding(encoding_scheme)
    
    def _initialize_network(self) -> None:
        """Initialize the Lightning BRNN network."""
        # Create network based on datatype
        if self.datatype == 'sequence':
            self.network = brnn_architecture.BRNN_MtO(**self.hyperparameters)
        elif self.datatype == 'residues':
            self.network = brnn_architecture.BRNN_MtM(**self.hyperparameters)
        else:
            raise ValueError(f"Invalid datatype: {self.datatype}")
        
        # Load the state dict
        self.network.load_state_dict(self.checkpoint['state_dict'])
        self.network.to(self.device)
        self.network.eval()
    
    def _postprocess_prediction(self, raw_output: torch.Tensor, sequence: str) -> np.ndarray:
        """Postprocess predictions for Lightning models."""
        prediction = raw_output.detach().cpu().numpy()
        
        if self.problem_type == 'classification':
            if self.datatype == 'residues':
                # For residue classification, apply softmax to each position
                prediction = prediction.reshape(-1, self.num_classes)
                prediction = np.array([self._softmax(row) for row in prediction])
            else:
                # For sequence classification, apply softmax to the output
                prediction = self._softmax(prediction.flatten())
        else:
            # For regression, just flatten the output
            prediction = prediction.flatten()
        
        return prediction
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax function to array."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)


def create_predictor(model_path: Union[str, Path], datatype: str = None, **kwargs) -> BasePredictor:
    """
    Factory function to create the appropriate predictor for a model.
    
    This function automatically detects the model type and creates the appropriate
    predictor class.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the saved model
    datatype : str, optional
        Data type for legacy models ('sequence' or 'residues'), by default None
    **kwargs
        Additional arguments passed to the predictor constructor
        
    Returns
    -------
    BasePredictor
        Appropriate predictor instance for the model
    """
    model_path = Path(model_path)
    
    # Load checkpoint to determine type
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    # Determine model type
    if 'hyper_parameters' in checkpoint:
        # Lightning checkpoint
        return LightningPredictor(model_path, **kwargs)
    else:
        # Legacy checkpoint
        if datatype is None:
            raise ValueError("datatype must be specified for legacy models")
        predictor = LegacyBRNNPredictor(model_path, **kwargs)
        predictor.datatype = datatype
        predictor._initialize_network()  # Re-initialize with correct datatype
        return predictor
