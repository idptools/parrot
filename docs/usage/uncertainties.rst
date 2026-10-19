=============================
 Values with uncertainties
=============================

Measured values are rarely equally reliable: some come from clean, well-replicated experiments and others from noisy ones. If each value in a regression datafile comes with an estimate of its measurement error, ``--uncertainties`` tells PARROT to take it into account. Precise values then pull the network harder than noisy ones, and PARROT also estimates how far its predictions are from the true values beyond those measurement errors, and checks that this account of the errors fits the test set.

``--uncertainties`` works for sequence and residue regression, with every architecture except ``knn`` (which cannot weight its training sequences) and ``unet2d`` (matrix data, which cannot have uncertainties yet). It is given to ``parrot-train``, ``parrot-optimize`` and ``parrot-preprocess``.

Datafile format
---------------

With ``--uncertainties``, every value is written as ``value,SD``, with no space, where SD is the standard deviation of that value's measurement error:

.. code-block:: none

    seq1  MKVLAAGIVGLLLA  3.21,0.15
    seq2  MDSKGSSQKGSRLL  0.84,0.40

For residue data, every residue's value has its own SD, and a residue with no value is still written as ``nan`` (it needs no SD):

.. code-block:: none

    seq1  MKVLA  0.81,0.05 0.77,0.04 nan 0.65,0.10 0.70,0.08

Each SD must be a finite number of at least 0. Every value needs one: a value without an SD is an error, not treated as exact. An SD of 0 is allowed, and means the value is exact (see "The model error" below for why this does not break anything). Without ``--uncertainties``, a ``value,SD`` token is an error that suggests the flag, so a datafile with SDs is never misread.

What the SD should be
---------------------

The SD should describe how far the value given could be from the true value, as one standard deviation of a Gaussian error. Which number that is depends on where the value comes from:

* **A single measurement or estimate with its own uncertainty** (an instrument's stated error, the standard error of a fitted parameter, the SD of a posterior distribution): use that SD as it is.
* **The mean of n replicates**: use the standard error of the mean, SD/√n, where SD is the spread between the replicates. The spread itself is the error of one replicate, not of their mean, and would make each value look √n times noisier than it is.
* **A 95% confidence interval**: use its half-width divided by 1.96.

How training uses the SDs
-------------------------

PARROT takes the difference between a value and its prediction to be the sum of two independent Gaussian errors: the value's measurement error, with the SD given, and the network's own error, with SD *s* (the "model error", the same for every value). So for a value *y* with SD σ and prediction μ,

.. math::

    y - \mu \sim \mathcal{N}(0,\ \sigma^2 + s^2)

and training minimises the negative log-likelihood of the values,

.. math::

    \mathrm{NLL} = \sum_i \tfrac{1}{2}\left[\log\left(2\pi(\sigma_i^2 + s^2)\right) + \frac{(y_i - \mu_i)^2}{\sigma_i^2 + s^2}\right]

in place of the usual loss (L1 for sequence regression, mean squared error for residue regression). In effect, each value is weighted by 1/(σ² + *s*\ ²): a miss of 2 on a value with SD 2 costs as much as a miss of 0.5 on a value with SD 0.5.

The model error
---------------

*s* is not a network weight. Before training, and after every epoch, it is set to its maximum-likelihood value given the network's residuals on the validation set, so it always matches the network as it is and describes how far its predictions are from the true values on sequences it was not trained on. (This is the likelihood estimate of the between-study variance of a random-effects meta-analysis, with the predictions in place of the pooled mean; Hardy & Thompson 1996, *Statistics in Medicine* 15:619-629.) The network kept, from the epoch with the lowest validation loss, keeps that epoch's *s*.

*s* is what stops a few very precise values from dominating training. Their weights are at most 1/*s*\ ²: once a value's SD is well below *s*, making it smaller does not make the value count for more, since the network cannot be trusted to that precision anyway. So:

* if the SDs are all much smaller than *s* (the network's own error swamps the measurement errors), every value counts about equally, as without ``--uncertainties``;
* if some SDs are much larger than *s*, those values count for much less;
* if the measurement errors alone explain the residuals, *s* is 0, and the values are weighted by 1/σ². (If some SDs are 0, *s* is kept just above 0, so no value has a variance of 0.)

PARROT prints *s* at the end of training (in the units of the values), and records it in "_info.txt" and "_performance_stats.txt". It is useful on its own: it says how much of the error is the network's rather than the measurements'. For a new sequence, the prediction is the network's best estimate of the true value, and *s* is a rough guide to how far off it may be (rough because it is one number for every sequence, and was measured on the validation set).

The simple models (``linear``, ``svm`` and ``random_forest``) are fitted with weights 1/(σ² + *s*\ ²) as scikit-learn sample weights, scaled to average 1 so that their regularisation keeps its strength. They are first fitted with equal weights, then refitted as *s* is re-estimated on the validation set, until it changes by less than 1% (at most five refits).

Results
-------

The test-set statistics (printed, and in "_performance_stats.txt" and "_info.txt") include, besides the usual correlations:

* **Reduced chi-squared**: the mean of (*y* − μ)²/(σ² + *s*\ ²) over the test values. About 1 if the account of the errors fits; well above 1 means the errors are larger than the SDs and *s* say (e.g. the SDs are underestimated, or the test set is harder than the validation set); well below 1 means they are smaller.
* **Within 1 SD** and **Within 2 SD**: the fraction of test values within 1 and 2 total SDs, √(σ² + *s*\ ²), of their prediction. About 0.68 and 0.95 if it fits.
* **Noise ceiling R**: the highest Pearson R any model could reach on these test values, given their measurement errors: √(1 − mean(σ²)/var(*y*)), assuming the errors are independent of the true values. Comparing it with the test Pearson R shows how much of the remaining error is the network's and how much the data's.

The predictions themselves are single values, as without ``--uncertainties``; "_predictions.tsv" gives the true values without their SDs (which are in the datafile).

Training details
----------------

* The validation loss is the same negative log-likelihood, so its values cannot be compared with those of a run without ``--uncertainties``, and it can be negative. ``--stop auto`` therefore asks for the validation loss to fall by about 0.005 per labelled value (rather than by 0.5%): the same 0.5% improvement, measured in likelihood.
* ``parrot-optimize`` scores configurations by the same validation loss, with *s* refitted for each configuration (and each cross-validation fold).
* ``parrot-preprocess --uncertainties --deduplicate merge`` combines the copies of a sequence (or of a residue) by their inverse-variance weighted mean, Σ(*y*/σ²)/Σ(1/σ²), with SD 1/√Σ(1/σ²): the standard way to combine independent estimates of one quantity. If a copy has an SD of 0, the merged value is the mean of the exact copies, with SD 0. Every labelled copy must have an SD that is a finite number of at least 0, or merging stops with a message naming the sequence.

Limitations
-----------

* The SDs describe random error, which weighting can average away. A systematic error (a bias) that depends on the sequence is a different matter: a network flexible enough to learn the values can learn the bias too, as if it were real, with or without ``--uncertainties``. The checks of the error model cannot reveal it either, since the test values carry the same bias. (In a simulation where hydrophilic sequences read too high, a ``brnn`` trained with ``--uncertainties`` reproduced most of the bias in its predictions, while its reduced chi-squared was 0.94 and 68% and 97% of test values were within 1 and 2 SDs.)
* The errors are taken to be Gaussian and independent. For residue data, if a whole sequence's values can be off together (e.g. one bad sample), its residues' errors are correlated, and the loss treats the sequence as more informative than it is. The weighting still helps, but is not exact.
* *s* is one number for every value. If the network is much better for some kinds of sequence than others, *s* averages over them.
* *s* is estimated from the validation set, so when it is much smaller than most of the SDs, a few hundred validation values pin it down only roughly. (In a simulation where most SDs were above 0.1, a ``linear`` model whose predictions were within 0.013 of the true values got *s* = 0.037 from 300 validation values; fitted to all 2,000 values, *s* was 0.010.)
* If how precisely a value is measured depends on the kind of sequence, the badly measured kinds count for little, so the network learns about them mostly from the others, and may need more epochs. (In a simulation where the SD doubled for every 0.4 that the mean hydropathy fell, a ``brnn`` trained with ``--uncertainties`` was no better than one trained without after 30 epochs, but had 13-36% lower error after 100, on hydrophilic and hydrophobic sequences alike.)
* Matrix data cannot have uncertainties yet.

An example
----------

On simulated data, 3,000 sequences whose true values are a fixed function of composition, measured with SDs from 0.05 to 3 (median 0.4) plus an extra error of SD 0.2 that no model could predict:

* the ``linear`` model trained with ``--uncertainties`` predicted the true values with an RMSE of 0.036, against 0.082 without; its *s* was 0.198 (the extra error was 0.2), the reduced chi-squared 1.03, and 66% and 94% of test values were within 1 and 2 total SDs;
* a ``brnn`` trained for 40 epochs, with three seeds (each its own random split), predicted the true values with an RMSE of 0.130, 0.142 and 0.155 with ``--uncertainties``, against 0.188, 0.221 and 0.291 without; *s* was 0.22-0.26 (the extra error plus the network's own), the reduced chi-squared 0.94-1.12, and 62-70% and 95-96% of test values were within 1 and 2 total SDs.

The Pearson R against the measured test values hardly changed (by at most 0.012, in favour of ``--uncertainties``), because those values are themselves noisy: their noise ceiling was 0.71-0.78. The improvement shows in how close the predictions are to the true values, which only a simulation can show.
