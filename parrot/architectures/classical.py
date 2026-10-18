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

With uncertainties (see parrot/uncertainties.py), each sequence is weighted by
1 / (SD^2 + s^2), where s is the model error: the linear, SVM and random-forest
models are fitted with these as scikit-learn sample weights, refitted as s is
re-estimated (see train_network.fit_in_one_step). k-nearest neighbours cannot
weight its training sequences, so it does not support uncertainties.

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
from collections.abc import Mapping
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

from parrot.architectures.base import ParrotArchitecture
from parrot.architectures.hyperparameters import HyperParameter, HyperParameterValue
from parrot.architectures.layers import (adjacent_pair_composition, composition_and_log_length, padding_mask,
                                         resolve_lengths, zero_padding)
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task
from parrot.uncertainties import MIN_VARIANCE, model_error_of

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

        Raises
        ------
        ValueError
            If the task is not sequence-mapped, a hyperparameter is missing,
            undeclared or not allowed, or 'dipeptides' features are asked for
            with an encoding of more than MAX_DIPEPTIDE_ENCODING_SIZE values per
            residue
        """
        super().__init__(input_size, task, hyperparameter_values)
        self.feature_set = str(hyperparameter_values['features'])
        if self.feature_set == 'dipeptides' and input_size > MAX_DIPEPTIDE_ENCODING_SIZE:
            raise ValueError(f"'dipeptides' features need an encoding of at most {MAX_DIPEPTIDE_ENCODING_SIZE} values per "
                             f'residue ({input_size}x{input_size} pairwise products would be too many); use '
                             "--features composition")
        self.estimator: Any = None

    @classmethod
    def unused_hyperparameters(cls, task: Task, input_size: int) -> dict[str, str]:
        """With a wide encoding only 'composition' features can be used (see ParrotArchitecture.unused_hyperparameters)."""
        if input_size <= MAX_DIPEPTIDE_ENCODING_SIZE:
            return {}
        return {'features': f"'dipeptides' needs an encoding of at most {MAX_DIPEPTIDE_ENCODING_SIZE} values per "
                            f'residue, and this one has {input_size}'}

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
        x = zero_padding(x, padding_mask(lengths, x.shape[1]))

        # The order [composition, log length, dipeptides] is what saved models were fitted to
        features = [composition_and_log_length(x, lengths)]
        if self.feature_set == 'dipeptides':
            features.append(adjacent_pair_composition(x, lengths))
        return torch.cat(features, dim=-1)

    def fit(self, loader: DataLoader[Any]) -> None:
        """Fit the model to every sequence in a dataset

        With uncertainties, each sequence is weighted by 1 / (SD^2 + s^2) (see
        sample_weights()), or equally while the model error s is not yet known.

        Parameters
        ----------
        loader : DataLoader
            Batches of the training data (process_input_data.Batch)
        """
        features, targets = [], []
        for batch in loader:
            features.append(self.sequence_features(batch.vectors, batch.lengths))
            targets.append(batch.targets)

        training_features = torch.cat(features).double().numpy()
        training_targets = torch.cat(targets).numpy()
        num_samples, num_features = training_features.shape

        pipeline = make_pipeline(StandardScaler(), self.make_estimator(num_samples, num_features))
        if self.task.is_classification:
            self.estimator = pipeline.fit(training_features, training_targets)
            return

        # Standard scaling can always be undone, so scikit-learn's check that it can is skipped: on single-precision
        # targets, rounding of values near 0 sometimes fails the check, with a spurious warning
        model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler(), check_inverse=False)
        fit_options = {}
        if self.task.uncertainties:
            # Targets are [value, SD] pairs (see process_input_data.pad_targets)
            weights = self.sample_weights(training_targets[:, 1])
            training_targets = training_targets[:, :1]
            if weights is not None:
                # The pipeline passes an option to the estimator that is its last step, named <step>__<option>
                fit_options[f'{pipeline.steps[-1][0]}__sample_weight'] = weights
        self.estimator = model.fit(training_features, training_targets.ravel(), **fit_options)

    def sample_weights(self, sds: np.ndarray) -> np.ndarray | None:
        """Weights for fitting to values with uncertainties: 1 / (SD^2 + s^2), scaled to average 1

        Scaling to an average of 1 keeps the regularisation (the ridge penalty,
        the SVM's C) as strong, relative to the fit, as it is without weights.

        Parameters
        ----------
        sds : numpy array
            The SD of each training value

        Returns
        -------
        numpy array or None
            None (equal weights) while the model error s is not yet known
        """
        model_error = float(model_error_of(self))
        if not np.isfinite(model_error):
            return None
        # As in the loss, the variance is kept above 0 (for an SD of 0 with a model error of exactly 0)
        weights = 1 / np.maximum(np.asarray(sds, dtype=np.float64) ** 2 + model_error ** 2, MIN_VARIANCE)
        normalised: np.ndarray = weights / weights.mean()
        return normalised

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
        HyperParameter('alpha', float, default=1.0, minimum=0, exclusive_minimum=True, search_range=(1e-3, 1e3),
                       log_scale=True, help='strength of the L2 penalty on the coefficients (must be > 0)'),
        FEATURES_HYPERPARAMETER,
    )

    def __init__(self, input_size: int, task: Task, alpha: float, features: str) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the model is for
        alpha : float
            Strength of the L2 penalty (> 0)
        features : str
            'composition' or 'dipeptides'
        """
        super().__init__(input_size, task, {'alpha': alpha, 'features': features})
        self.alpha = alpha

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        """Ridge regression, or logistic regression for classification (see ClassicalModel.make_estimator)."""
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
        HyperParameter('svm_c', float, default=1.0, minimum=0, exclusive_minimum=True, search_range=(1e-2, 1e3),
                       log_scale=True,
                       help='penalty on training errors (scikit-learn C, must be > 0); larger values fit the '
                            'training data more closely'),
        HyperParameter('kernel', str, default='rbf', choices=('rbf', 'linear', 'poly'), search_choices=('rbf', 'linear'),
                       help='kernel function'),
        HyperParameter('relative_gamma', float, default=1.0, minimum=0, exclusive_minimum=True,
                       search_range=(1e-2, 1e2), log_scale=True,
                       help="kernel coefficient as a multiple of scikit-learn's default (1 / number of features); "
                            "larger values give more local, wigglier fits; not used by the linear kernel"),
        FEATURES_HYPERPARAMETER,
    )

    def __init__(self, input_size: int, task: Task, svm_c: float, kernel: str, relative_gamma: float,
                 features: str) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the model is for
        svm_c : float
            Penalty on training errors (> 0)
        kernel : str
            'rbf', 'linear' or 'poly'
        relative_gamma : float
            Kernel coefficient as a multiple of 1 / number of features (> 0)
        features : str
            'composition' or 'dipeptides'
        """
        super().__init__(input_size, task, {'svm_c': svm_c, 'kernel': kernel, 'relative_gamma': relative_gamma,
                                            'features': features})
        self.svm_c = svm_c
        self.kernel = kernel
        self.relative_gamma = relative_gamma

    @classmethod
    def inactive_hyperparameters(cls, values: Mapping[str, HyperParameterValue]) -> set[str]:
        """The linear kernel has no coefficient (see ParrotArchitecture.inactive_hyperparameters)."""
        return {'relative_gamma'} if values.get('kernel') == 'linear' else set()

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        """SVR, or SVC with class probabilities for classification (see ClassicalModel.make_estimator)."""
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
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the model is for
        num_trees : int
            Number of trees
        min_samples_leaf : int
            Minimum number of training sequences in each leaf
        features : str
            'composition' or 'dipeptides'
        """
        super().__init__(input_size, task, {'num_trees': num_trees, 'min_samples_leaf': min_samples_leaf,
                                            'features': features})
        self.num_trees = num_trees
        self.min_samples_leaf = min_samples_leaf

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        """A random forest classifier or regressor (see ClassicalModel.make_estimator)."""
        forest = RandomForestClassifier if self.task.is_classification else RandomForestRegressor
        return forest(n_estimators=self.num_trees, min_samples_leaf=self.min_samples_leaf,
                      random_state=random_state())


@register_architecture
class KNearestNeighbors(ClassicalModel):
    """Predicts from the most similar training sequences (in standardised feature space)"""

    name = 'knn'
    description = 'k-nearest neighbours on sequence features'
    # scikit-learn's k-nearest neighbours cannot weight its training sequences
    supports_uncertainties = False
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
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the model is for
        num_neighbors : int
            Number of most similar training sequences used for each prediction
        neighbor_weights : str
            'uniform' or 'distance'
        features : str
            'composition' or 'dipeptides'
        """
        super().__init__(input_size, task, {'num_neighbors': num_neighbors, 'neighbor_weights': neighbor_weights,
                                            'features': features})
        self.num_neighbors = num_neighbors
        self.neighbor_weights = neighbor_weights

    def make_estimator(self, num_samples: int, num_features: int) -> Any:
        """A k-nearest-neighbours classifier or regressor (see ClassicalModel.make_estimator).

        Raises
        ------
        ValueError
            If there are fewer training sequences than neighbours
        """
        if self.num_neighbors > num_samples:
            raise ValueError(f'num_neighbors ({self.num_neighbors}) is larger than the number of training '
                             f'sequences ({num_samples})')

        neighbours = KNeighborsClassifier if self.task.is_classification else KNeighborsRegressor
        return neighbours(n_neighbors=self.num_neighbors, weights=self.neighbor_weights)
