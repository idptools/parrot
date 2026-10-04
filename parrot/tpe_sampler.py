"""
Suggesting configurations to try with a tree-structured Parzen estimator (TPE;
Bergstra et al. 2011, NeurIPS), in the multivariate form used by BOHB (Falkner
et al. 2018, ICML).

The configurations scored at one stage of the search are split into the best
GOOD_FRACTION ("good") and the rest ("bad"), and a kernel density estimate is
fitted to each. New candidates are drawn from the good density, and the one with
the highest ratio of good to bad density is tried next: it looks like the good
configurations and unlike the bad ones.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import math

import numpy as np
from scipy.special import logsumexp, ndtr

from parrot.search_space import SearchSpace

# TPE settings. The first three are BOHB's defaults (Falkner et al. 2018: top_n_percent, num_samples and
# bandwidth_factor in its hpbandster implementation).
GOOD_FRACTION = 0.15            # fraction of scored configurations whose density counts as "good" (at least
                                # one more than the number of hyperparameters)
CANDIDATES_PER_SUGGESTION = 64  # candidates drawn from the good density for each new configuration
BANDWIDTH_FACTOR = 3.0          # candidates are drawn from the good density with its kernels widened this much

# PARROT's own TPE settings (BOHB uses a smaller minimum bandwidth, and estimates the spread of choices from
# the data): kernels never narrower than 5% of a number's range, and a fixed 20% chance that a choice's
# kernel spreads to the other choices, so the model never rules out any part of the space.
MIN_BANDWIDTH = 0.05            # narrowest kernel for a number, as a fraction of its search range
CATEGORY_SMOOTHING = 0.2        # probability a choice's kernel spreads over the other choices

# A candidate number drawn outside [0, 1] is redrawn up to this many times, then clipped
MAX_REDRAWS = 20


class TPESampler:
    """Suggests configurations where good results have clustered (tree-structured Parzen estimator)

    Each density is a kernel density estimate over points: the average of one
    kernel per point, where a point's kernel is the product of one kernel per
    coordinate. A number's kernel is a normal distribution centred on the
    point's coordinate and truncated to [0, 1]; a choice's kernel gives the
    point's own choice probability 1 - CATEGORY_SMOOTHING and shares
    CATEGORY_SMOOTHING equally between the other choices.

    Attributes
    ----------
    space : SearchSpace
    rng : numpy.random.Generator
    """

    def __init__(self, space: SearchSpace, rng: np.random.Generator) -> None:
        """
        Parameters
        ----------
        space : SearchSpace
            The space being searched
        rng : numpy.random.Generator
            Source of randomness for drawing candidates
        """
        self.space = space
        self.rng = rng
        self._numeric = np.array([space.is_numeric(hyperparameter) for hyperparameter in space.dimensions])
        self._choice_counts = np.array([len(hyperparameter.search_choices or ()) for hyperparameter in space.dimensions])

    def n_good(self, n_observations: int) -> int:
        """How many of the best configurations make up the good density.

        Parameters
        ----------
        n_observations : int
            Number of configurations scored at the stage being modelled

        Returns
        -------
        int
        """
        return max(len(self.space) + 1, math.ceil(GOOD_FRACTION * n_observations))

    def model_can_be_fitted(self, n_observations: int) -> bool:
        """Whether there are enough scored configurations at one stage to fit the densities

        As in BOHB, both the good and the bad density need at least one more
        configuration than there are hyperparameters, so neither collapses onto a
        few points.

        Parameters
        ----------
        n_observations : int
            Number of configurations scored at the stage

        Returns
        -------
        bool
        """
        return n_observations - self.n_good(n_observations) >= len(self.space) + 1

    def suggest(self, points: np.ndarray, losses: np.ndarray) -> np.ndarray:
        """Candidate points drawn near the good configurations, best first

        Candidates are drawn from the good density (with its kernels widened by
        BANDWIDTH_FACTOR, to explore around the good configurations) and ranked
        by the ratio of the good to the bad density at each.

        Parameters
        ----------
        points : numpy array
            [n X dimensions] points scored at one stage
        losses : numpy array
            Their losses (inf for failed configurations)

        Returns
        -------
        numpy array
            [CANDIDATES_PER_SUGGESTION X dimensions] candidates, best first
        """
        n_good = self.n_good(len(points))
        order = np.argsort(losses, kind='stable')
        good, bad = points[order[:n_good]], points[order[n_good:]]

        good_bandwidth, bad_bandwidth = self._bandwidths(good), self._bandwidths(bad)
        candidates = self._sample(good, good_bandwidth * BANDWIDTH_FACTOR, CANDIDATES_PER_SUGGESTION)
        log_ratio = self._log_density(candidates, good, good_bandwidth) - self._log_density(candidates, bad, bad_bandwidth)
        return candidates[np.argsort(-log_ratio, kind='stable')]

    def _bandwidths(self, points: np.ndarray) -> np.ndarray:
        """Kernel width for each numeric coordinate (Scott's rule), clipped to [MIN_BANDWIDTH, 1].

        Parameters
        ----------
        points : numpy array
            [n X dimensions] points the density is fitted to

        Returns
        -------
        numpy array
            One width per dimension (only used for numeric dimensions)
        """
        # Scott's rule for a d-dimensional kernel density estimate: standard deviation x n^(-1 / (d + 4))
        scott = points.std(axis=0) * len(points) ** (-1.0 / (len(self.space) + 4))
        return np.clip(scott, MIN_BANDWIDTH, 1.0)

    def _sample(self, centres: np.ndarray, bandwidth: np.ndarray, n: int) -> np.ndarray:
        """Draw n points from the kernel density centred on `centres`.

        Parameters
        ----------
        centres : numpy array
            [n_centres X dimensions] points the density is made of
        bandwidth : numpy array
            Kernel width for each numeric dimension
        n : int
            Number of points to draw

        Returns
        -------
        numpy array
            [n X dimensions]
        """
        # Pick a centre for each sample, then move each coordinate by its kernel
        samples = centres[self.rng.integers(len(centres), size=n)].copy()
        for dimension in range(len(self.space)):
            if self._numeric[dimension]:
                samples[:, dimension] = self._truncated_normal(samples[:, dimension], bandwidth[dimension])
                continue

            # A choice moves to one of the other choices with probability CATEGORY_SMOOTHING, widened by
            # BANDWIDTH_FACTOR like the numbers, but never above (count - 1) / count, where every choice is
            # equally likely (beyond it, the kernel would favour the other choices over its own)
            count = self._choice_counts[dimension]
            switch = self.rng.random(n) < min(1 - 1 / count, CATEGORY_SMOOTHING * BANDWIDTH_FACTOR)
            offsets = self.rng.integers(1, count, size=n)
            samples[switch, dimension] = (samples[switch, dimension] + offsets[switch]) % count
        return samples

    def _truncated_normal(self, means: np.ndarray, width: float) -> np.ndarray:
        """Normal samples around `means`, redrawn until they fall in [0, 1] (clipped as a last resort).

        Parameters
        ----------
        means : numpy array
            Centre of each sample
        width : float
            Standard deviation

        Returns
        -------
        numpy array
            One sample per mean, in [0, 1]
        """
        values = self.rng.normal(means, width)
        for _ in range(MAX_REDRAWS):
            outside = (values < 0) | (values > 1)
            if not outside.any():
                break
            values[outside] = self.rng.normal(means[outside], width)
        return np.clip(values, 0, 1)

    def _log_density(self, points: np.ndarray, centres: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """Log of the kernel density estimate centred on `centres` (see the class docstring), at each point.

        Parameters
        ----------
        points : numpy array
            [n_points X dimensions] where to evaluate the density
        centres : numpy array
            [n_centres X dimensions] points the density is made of
        bandwidth : numpy array
            Kernel width for each numeric dimension

        Returns
        -------
        numpy array
            [n_points] log densities
        """
        # log_kernels[i, j]: log of centre j's kernel at point i, built up one dimension at a time
        log_kernels = np.zeros((len(points), len(centres)))
        for dimension in range(len(self.space)):
            point_coordinates = points[:, dimension][:, None]
            centre_coordinates = centres[:, dimension][None, :]

            if self._numeric[dimension]:
                # Log of a normal density, divided by the fraction of the normal that lies in [0, 1]
                width = bandwidth[dimension]
                z = (point_coordinates - centre_coordinates) / width
                mass_inside = ndtr((1 - centre_coordinates) / width) - ndtr(-centre_coordinates / width)
                log_kernels += -0.5 * z ** 2 - math.log(width * math.sqrt(2 * math.pi)) - np.log(mass_inside)
            else:
                count = self._choice_counts[dimension]
                same_choice = np.isclose(point_coordinates, centre_coordinates)
                log_kernels += np.where(same_choice, math.log(1 - CATEGORY_SMOOTHING),
                                        math.log(CATEGORY_SMOOTHING / (count - 1)))

        # The density is the average of the kernels: log(mean(exp(log_kernels))) over centres
        return logsumexp(log_kernels, axis=1) - math.log(len(centres))
