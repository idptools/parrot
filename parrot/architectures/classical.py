"""
Simple, non-neural models for sequence-mapped data, built on scikit-learn.

Each sequence is summarised by a fixed set of features: its composition (the
average encoded residue; for one-hot encoding, the fraction of each amino acid)
and the log of its length, optionally with its dipeptide composition. A classical
model is then fitted to these features in a single step. These models train in
seconds, have few hyperparameters and are useful baselines: a neural network
that cannot beat them is not learning much from sequence order.

Features are standardised before fitting, and so are regression targets (the
standardisation is undone when predicting), so hyperparameters such as the SVM
penalty behave consistently across datasets.

The models behave like any other PARROT architecture: forward() maps a padded
batch to outputs (for classification, the log of the predicted class
probabilities, which act as logits), so evaluation, prediction, reporting and
saving all work unchanged. train_network.train() calls fit() instead of running
gradient descent.

Saved networks store the fitted scikit-learn model as a pickle inside the PARROT
file. Unpickling can run arbitrary code, so only load classical models from
sources you trust.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import pickle
from typing import Any

import numpy as np
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader

from parrot.architectures.base import HyperParameter, HyperParameterValue, ParrotArchitecture
from parrot.architectures.layers import adjacent_pair_composition, mean_over_residues, padding_mask, resolve_lengths
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task

# Sets of sequence features the models can use
FEATURE_SETS = ('composition', 'dipeptides')

# Class probabilities are clipped to at least this before taking logs, so a class the model
# never predicts (e.g. one missing from a cross-validation fold) gets a large negative logit
# rather than an infinite one
MINIMUM_PROBABILITY = 1e-7

# Maximum solver iterations for logistic regression (scikit-learn's default of 100 often
# stops before convergence)
LOGISTIC_MAX_ITERATIONS = 1000

# Exclusive upper bound for the random seeds passed to scikit-learn
MAX_RANDOM_STATE = 2**31 - 1

# Dipeptide features are every pairwise product of the encoding's values, so their
# number grows as the square of the encoding size. Beyond this size (e.g. for protein
# language model embeddings, with hundreds of values per residue) they would need too
# much memory, so only 'composition' features are allowed.
MAX_DIPEPTIDE_ENCODING_SIZE = 32

FEATURES_HYPERPARAMETER = HyperParameter(
    'features', str, default='composition', choices=FEATURE_SETS, search_choices=FEATURE_SETS,
    help="sequence features: 'composition' (composition and log length) or 'dipeptides' (also dipeptide composition)")


def random_state() -> int:
    """A seed for scikit-learn, drawn from NumPy's global generator so --seed makes fitting reproducible."""
    return int(np.random.randint(MAX_RANDOM_STATE))


def require_positive(name: str, value: float) -> None:
    """Raise a ValueError unless `value` is greater than zero."""
    if value <= 0:
        raise ValueError(f"Hyperparameter '{name}' must be greater than 0, got {value}")


class ClassicalModel(ParrotArchitecture):
    """Base class for models fitted in one step to fixed sequence features

    Subclasses declare their hyperparameters (including FEATURES_HYPERPARAMETER)
    and implement make_estimator().

    Attributes
    ----------
    feature_set : str
        'composition' or 'dipeptides'
    estimator : scikit-learn estimator or None
        The fitted model (feature scaling included), or None before fitting
    """

    supported_datatypes = ('sequence',)
    trained_by_gradient_descent = False

    def __init__(self, input_size: int, task: Task, hyperparameter_values: dict[str, HyperParameterValue]) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the model is for (must be sequence-mapped)
        hyperparameter_values : dict
            The value of every declared hyperparameter, including 'features'
        """
        super().__init__(input_size, task, hyperparameter_values)
        self.feature_set = str(hyperparameter_values['features'])
        if self.feature_set == 'dipeptides' and input_size > MAX_DIPEPTIDE_ENCODING_SIZE:
            raise ValueError(f"'dipeptides' features need an encoding of at most {MAX_DIPEPTIDE_ENCODING_SIZE} values per "
                             f'residue ({input_size}x{input_size} pairwise products would be too many); use '
                             "--features composition")
        self.estimator: Any = None

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        """Build the (unfitted) scikit-learn estimator for this model and task

        Parameters
        ----------
        num_samples : int
            Number of training sequences
        num_features : int
            Number of features per sequence

        Returns
        -------
        scikit-learn estimator
        """
        raise NotImplementedError

    def sequence_features(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Summarise each sequence in a padded batch as a fixed-length feature vector

        Parameters
        ----------
        x : torch.Tensor
            Zero-padded batch [batch_dim X padded_length X input_size]
        lengths : torch.Tensor or None
            The true length of each sequence (None if the batch is unpadded)

        Returns
        -------
        torch.Tensor
            [batch_dim X num_features]: the mean encoded residue and the log of the
            length, followed (for 'dipeptides') by the adjacent-pair composition
        """
        lengths = resolve_lengths(x, lengths)
        x = x * padding_mask(lengths, x.shape[1]).unsqueeze(-1).to(x.dtype)

        features = [mean_over_residues(x, lengths), torch.log(lengths.to(x.dtype)).unsqueeze(-1)]
        if self.feature_set == 'dipeptides':
            features.append(adjacent_pair_composition(x, lengths))
        return torch.cat(features, dim=-1)

    def fit(self, loader: DataLoader[Any]) -> None:
        """Fit the model to every sequence in a dataset

        Parameters
        ----------
        loader : DataLoader
            Batches of the training data
        """
        features, targets = [], []
        for _, vectors, batch_targets, lengths in loader:
            features.append(self.sequence_features(vectors, lengths))
            targets.append(batch_targets)

        training_features = torch.cat(features).double().numpy()
        training_targets = torch.cat(targets).numpy()
        num_samples, num_features = training_features.shape

        model = make_pipeline(StandardScaler(), self.make_estimator(num_samples, num_features))
        if not self.task.is_classification:
            model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
            training_targets = training_targets.ravel()

        self.estimator = model.fit(training_features, training_targets)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Predict from sequence features

        Parameters
        ----------
        x : torch.Tensor
            Zero-padded batch [batch_dim X padded_length X input_size]
        lengths : torch.Tensor, optional
            The true length of each sequence (None if the batch is unpadded)

        Returns
        -------
        torch.Tensor
            [batch_dim X num_classes]: predicted values for regression, or log class
            probabilities for classification

        Raises
        ------
        RuntimeError
            If the model has not been fitted
        """
        if self.estimator is None:
            raise RuntimeError(f"The '{self.name}' model has not been fitted yet")

        features = self.sequence_features(x, lengths).detach().cpu().double().numpy()
        if self.task.is_classification:
            outputs = self.class_log_probabilities(features)
        else:
            outputs = self.estimator.predict(features).reshape(-1, 1)

        return torch.as_tensor(outputs, dtype=x.dtype, device=x.device)

    def class_log_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Log probability of every class, including classes absent from the training data

        Parameters
        ----------
        features : numpy array
            [num_sequences X num_features]

        Returns
        -------
        numpy array
            [num_sequences X num_classes]
        """
        probabilities = np.zeros((len(features), self.num_classes))
        probabilities[:, self.estimator.classes_] = self.estimator.predict_proba(features)
        log_probabilities: np.ndarray = np.log(np.clip(probabilities, MINIMUM_PROBABILITY, None))
        return log_probabilities

    def get_extra_state(self) -> bytes | None:
        """Store the fitted estimator in the network's state_dict (and so in saved networks)."""
        if self.estimator is None:
            return None
        return pickle.dumps(self.estimator)

    def set_extra_state(self, state: bytes | None) -> None:
        """Restore the fitted estimator from a state_dict (see the warning in the module docstring)."""
        self.estimator = None if state is None else pickle.loads(state)


@register_architecture
class LinearModel(ClassicalModel):
    """Ridge regression (regression) or L2-penalised logistic regression (classification)"""

    name = 'linear'
    description = 'ridge or logistic regression on sequence features'
    hyperparameters = (
        HyperParameter('alpha', float, default=1.0, search_range=(1e-3, 1e3), log_scale=True,
                       help='strength of the L2 penalty on the coefficients (must be > 0)'),
        FEATURES_HYPERPARAMETER,
    )

    def __init__(self, input_size: int, task: Task, alpha: float, features: str) -> None:
        require_positive('alpha', alpha)
        super().__init__(input_size, task, {'alpha': alpha, 'features': features})
        self.alpha = alpha

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        if not self.task.is_classification:
            return Ridge(alpha=self.alpha)

        # scikit-learn's C is the inverse of the penalty strength
        return LogisticRegression(C=1 / self.alpha, max_iter=LOGISTIC_MAX_ITERATIONS)


@register_architecture
class SupportVectorMachine(ClassicalModel):
    """Support vector regression (SVR) or classification (SVC)"""

    name = 'svm'
    description = 'support vector machine on sequence features'
    hyperparameters = (
        HyperParameter('svm_c', float, default=1.0, search_range=(1e-2, 1e3), log_scale=True,
                       help='penalty on training errors (scikit-learn C, must be > 0); larger values fit the '
                            'training data more closely'),
        HyperParameter('kernel', str, default='rbf', choices=('rbf', 'linear', 'poly'), search_choices=('rbf', 'linear'),
                       help='kernel function'),
        HyperParameter('relative_gamma', float, default=1.0, search_range=(1e-2, 1e2), log_scale=True,
                       help="kernel coefficient as a multiple of scikit-learn's default (1 / number of features); "
                            "larger values give more local, wigglier fits; not used by the linear kernel"),
        FEATURES_HYPERPARAMETER,
    )

    def __init__(self, input_size: int, task: Task, svm_c: float, kernel: str, relative_gamma: float,
                 features: str) -> None:
        require_positive('svm_c', svm_c)
        require_positive('relative_gamma', relative_gamma)
        super().__init__(input_size, task, {'svm_c': svm_c, 'kernel': kernel, 'relative_gamma': relative_gamma,
                                            'features': features})
        self.svm_c = svm_c
        self.kernel = kernel
        self.relative_gamma = relative_gamma

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        # For standardised features scikit-learn's default gamma ('scale') is about 1 / num_features
        gamma = self.relative_gamma / num_features

        if self.task.is_classification:
            # probability=True fits the calibration needed for class probabilities
            return SVC(C=self.svm_c, kernel=self.kernel, gamma=gamma, probability=True,
                       random_state=random_state())
        return SVR(C=self.svm_c, kernel=self.kernel, gamma=gamma)


@register_architecture
class RandomForest(ClassicalModel):
    """Random forest of decision trees"""

    name = 'random_forest'
    description = 'random forest on sequence features'
    hyperparameters = (
        HyperParameter('num_trees', int, default=200, minimum=1, search_range=(50, 500), help='number of trees'),
        HyperParameter('min_samples_leaf', int, default=1, minimum=1, search_range=(1, 20),
                       help='minimum number of training sequences in each leaf; larger values give smoother predictions'),
        FEATURES_HYPERPARAMETER,
    )

    def __init__(self, input_size: int, task: Task, num_trees: int, min_samples_leaf: int, features: str) -> None:
        super().__init__(input_size, task, {'num_trees': num_trees, 'min_samples_leaf': min_samples_leaf,
                                            'features': features})
        self.num_trees = num_trees
        self.min_samples_leaf = min_samples_leaf

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        forest = RandomForestClassifier if self.task.is_classification else RandomForestRegressor
        return forest(n_estimators=self.num_trees, min_samples_leaf=self.min_samples_leaf,
                      random_state=random_state())


@register_architecture
class KNearestNeighbors(ClassicalModel):
    """Predicts from the most similar training sequences (in standardised feature space)"""

    name = 'knn'
    description = 'k-nearest neighbours on sequence features'
    hyperparameters = (
        HyperParameter('num_neighbors', int, default=5, minimum=1, search_range=(1, 50),
                       help='number of most similar training sequences used for each prediction'),
        HyperParameter('neighbor_weights', str, default='uniform', choices=('uniform', 'distance'),
                       search_choices=('uniform', 'distance'),
                       help="'uniform' (each neighbour counts equally) or 'distance' (closer neighbours count more)"),
        FEATURES_HYPERPARAMETER,
    )

    def __init__(self, input_size: int, task: Task, num_neighbors: int, neighbor_weights: str,
                 features: str) -> None:
        super().__init__(input_size, task, {'num_neighbors': num_neighbors, 'neighbor_weights': neighbor_weights,
                                            'features': features})
        self.num_neighbors = num_neighbors
        self.neighbor_weights = neighbor_weights

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        if self.num_neighbors > num_samples:
            raise ValueError(f'num_neighbors ({self.num_neighbors}) is larger than the number of training '
                             f'sequences ({num_samples})')

        neighbours = KNeighborsClassifier if self.task.is_classification else KNeighborsRegressor
        return neighbours(n_neighbors=self.num_neighbors, weights=self.neighbor_weights)
