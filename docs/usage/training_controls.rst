.. _parrot-training-controls:

===================
 Training controls
===================

Neural networks trained by ``parrot-train`` and ``parrot-optimize`` share the
controls on this page, including ``unet2d`` for matrix data. Every control is
optional. By default, PARROT uses Adam with a fixed user-specified learning rate,
sums the labelled losses, and does not clip gradients or retry numerical failures.
The simple models, which are fitted in one step, do not offer these options.

Checkpointing, continuous logs, loss selection, regularization and CUDA mixed
precision are also opt-in; they are described below. The command references list the flags (:doc:`parrot-train` and
:doc:`parrot-optimize`); :doc:`advanced_examples` gives complete commands, and
:doc:`../api` shows how to use the controls from Python.

.. _parrot-loss-weighting:

Residue weighting and equal-protein weighting
---------------------------------------------

For residue problems, the default loss sums labelled residue losses, so a protein
with more labelled residues contributes more terms. This is a natural objective
when each labelled residue should count equally: a protein with 1,000 measurements
contributes ten times as many loss terms as one with 100. Different protein
lengths alone do not require equal-protein weighting.

``--equal-protein-weighting`` instead averages labelled losses within each protein,
then averages those protein means for the batch's update. The two proteins then
have equal objective weight. Their gradients can still differ in magnitude and
direction; this does not force equal gradients. Choose this option when the
objective is the average protein's prediction error, regardless of how many
labels each protein has. It applies to regression and classification.

Padding and missing labels are excluded from the counts. For matrix tasks, each
protein's loss is averaged over its labelled cells; with ``--symmetric``, only
the diagonal and upper triangle count, as in the default loss. Sequence tasks
already have one labelled value per protein and do not accept this flag.

The same weighting is used for training, validation, test loss, early stopping
and hyperparameter selection. With :doc:`uncertainties`, both the Gaussian loss
and model-error fitting use the per-protein weights. Standard performance metrics,
including correlations, accuracy and uncertainty diagnostics, retain their usual
aggregation over labelled values.

Reported losses are averaged over the number of proteins in either mode: by
default, the mean of the proteins' summed losses; with this option, the mean of
their mean labelled losses. Thus the loss values and gradient scale change when
the flag is enabled. Recalibrate learning rate and any clipping threshold, and
compare validation losses only between runs using the same objective.

.. _parrot-regression-losses:

Regression losses and regularization
------------------------------------

``--regression-loss mae``, ``mse`` or ``huber`` selects the neural network's
regression objective independently of datatype. The defaults remain MAE for
sequence regression and MSE for residue and matrix regression. Huber uses
squared error near zero and a linear penalty for larger errors;
``--huber-delta`` sets its positive, finite transition in target units (default
1). Padding, missing labels and symmetric matrix cells follow the same masking
rules for every loss, including with equal-protein weighting.

The selected loss is used for training, validation, test loss and hyperparameter
selection, and saved with the network. It cannot be combined with classification
or ``--uncertainties``, whose objective remains Gaussian negative log-likelihood.
Simple models fitted in one step do not offer these loss flags.

``--optimizer adam`` retains the existing optimizer. ``--optimizer adamw`` opts
in to AdamW's decoupled weight decay. ``--weight-decay`` is finite and nonnegative
and defaults to 0 for both optimizers; choosing AdamW does not implicitly enable
regularization. Adam's weight decay is an L2 penalty within its adaptive update,
while AdamW applies decay separately. Compare these settings on validation data,
and record the choice with the loss and learning rate. They are fixed policies
throughout ``parrot-optimize``, including final retraining.

.. _parrot-lr-schedules:

Learning-rate schedules and warmup
----------------------------------

``--learning-rate`` / ``-lr`` gives the initial learning rate (default 0.001 for
``parrot-train``). ``--lr-schedule fixed`` keeps it constant, the default.
Adam already adapts individual parameter updates from gradient statistics;
the schedules below additionally change the overall learning rate over epochs.

* ``--lr-schedule cosine`` smoothly anneals toward ``--lr-min`` (default 0).
  ``--lr-anneal-epochs`` sets the decay duration after warmup; by default it is
  the full run's ``--epochs`` minus warmup, with a minimum of 1. The first decay
  epoch uses the initial rate. After that many completed decay epochs, the rate
  reaches the floor and stays there, without starting another cycle. With
  ``--stop auto``, whose run length is unknown, an explicit duration is required.
* ``--lr-schedule exponential`` multiplies the initial rate by ``--lr-gamma``
  (default 0.95) for each completed decay epoch, down to the floor.
* ``--lr-schedule plateau`` adapts the rate to validation loss. It multiplies
  the rate by ``--lr-factor`` (default 0.5) after the number of epochs without
  sufficient improvement exceeds ``--lr-patience`` (default 5). For example,
  patience 5 tolerates five such epochs and reduces after the sixth.
  ``--lr-threshold`` sets the minimum absolute improvement in validation-loss
  units (default 0.0001), so it also works with negative Gaussian losses.
  ``--lr-cooldown`` sets the cooldown after a reduction (default 0).
  Only validation data controls these reductions.

``--lr-warmup-epochs N`` optionally ramps the rate linearly from initial-rate/N
to the initial rate over N epochs before the chosen schedule, including a fixed
rate. The first epoch after warmup uses the initial rate again; subsequent
epochs follow the decay schedule. Plateau monitoring begins after warmup.

``--lr-min`` is a lower bound during warmup, decay and recovery, and must be
finite, nonnegative and no greater than the initial rate. Reduction factors
(``--lr-gamma`` and ``--lr-factor``) must be strictly between 0 and 1. Schedule
tuning flags are accepted only when they apply: for example, ``--lr-gamma``
requires ``--lr-schedule exponential``. ``--lr-min`` requires a schedule,
warmup or recovery.

With ``--stop auto``, choose stopping patience (``--epochs``) long enough to let
plateau reductions take effect; for example, ``--lr-patience 3 --epochs 10``.
Early stopping has its own improvement criterion and does not reset when the
learning rate changes. For losses with uncertainties and equal-protein
weighting, its improvement threshold is about 0.005 in the reported mean
per-protein NLL, rather than 0.005 per labelled value.

.. _parrot-gradient-safeguards:

Gradient clipping and finite checks
-----------------------------------

Two optional safeguards check for NaN or infinite values:

* ``--check-finite-gradients`` stops if a training batch's loss or global L2
  gradient norm is nonfinite, before updating Adam for that batch. It also
  stops on a nonfinite validation loss. It does not clip gradients.
* ``--gradient-clip-norm MAX_NORM`` scales the gradients together if their
  global L2 norm exceeds this positive, finite threshold. It also enables
  the finite checks: clipping cannot repair gradients that are already NaN
  or infinite.

With either option, ``parrot-train`` reports the median, 95th percentile and
maximum pre-clipping gradient norm for epochs whose progress is printed
(``--verbose`` prints every epoch). With clipping it also reports the fraction
of batches exceeding the threshold. ``--silent`` suppresses these summaries.

There is no universal clipping threshold: its scale depends on the architecture,
target values, batch size, sequence lengths and loss reduction. To calibrate one,
run a short pilot at a stable learning rate with
``--check-finite-gradients --verbose`` and inspect the norm distribution. Then
compare validation performance and clipping frequency across pilots with
candidate thresholds. Reconsider the threshold if you change those settings,
especially ``--equal-protein-weighting``. The safeguards themselves do not
change loss weighting.

Training failures identify the epoch, batch, representative sequence IDs and
length range. Without recovery enabled, the run stops without skipping the batch
or retrying. Any earlier best network saved to disk is preserved.

.. _parrot-numerical-recovery:

Bounded recovery from numerical failures
----------------------------------------

``--recover-nonfinite`` enables at most two rollbacks per training run. Give a
positive integer to change the budget, e.g. ``--recover-nonfinite 1``; 0 disables
it. The budget never resets after successful epochs.

At each epoch boundary, PARROT keeps a clean CPU snapshot of model parameters
and buffers, Adam state, learning-rate control and Python/NumPy/PyTorch random
states, including the active CUDA or MPS generator and explicit loader generators.
On a numerical failure, it restores that boundary and replays the entire epoch
at a lower rate. ``--recovery-lr-factor`` (default 0.5, strictly between 0 and 1)
controls the reduction. Repeated failures reduce the rate cumulatively; future
scheduling retains the reduction and respects ``--lr-min``. Failed attempts
never update the saved best network, successful-epoch history or plateau state.

Recovery enables checks of inputs, loss, gradients, model tensors and optimizer
moments. Invalid inputs or targets stop immediately and are not retried; the
supported missing-label markers are still allowed. Arbitrary runtime errors
are not retried. If the budget is exhausted or the rate cannot be reduced further,
PARROT restores the clean boundary, stops and preserves any earlier saved best
network. Reducing the learning rate cannot fix every failure: for example,
overflow during the first backward pass can recur before a smaller update has
had a chance to help.

Recovery is within the running process, and its snapshots consume CPU memory.
For restarting after a process or machine crash, enable the separate disk
checkpoints below. The saved network remains an inference checkpoint.
Reproducible rollback requires loaders with ``num_workers=0``, as used
by PARROT's CLI; custom multiworker loaders are rejected when recovery is enabled.

.. _parrot-training-resume:

Disk checkpoints and resume
---------------------------

``--checkpoint`` enables resumable disk state, written atomically to
``<network>_training_checkpoint.pt``. An optional filename chooses another path.
``--checkpoint-every N`` sets the interval in successful epochs (default 1).
An initial clean checkpoint is written before training, and the last epoch and
each recovery rollback are also checkpointed. Existing checkpoint files are
not overwritten by a new run; resume them or choose a new path.

To continue an interrupted run, the checkpoint supplies the original arguments:

.. code-block:: bash

    parrot-train --resume network_training_checkpoint.pt
    parrot-optimize --resume optimized_training_checkpoint.pt

The supplied resume file also receives subsequent checkpoints. This works after
copying or renaming a checkpoint or changing the working directory; the original
checkpoint path is not restored. To write subsequent checkpoints elsewhere, give
an explicit ``--checkpoint <new-file>`` alongside ``--resume``. Input files must
still be available at their saved paths.

Checkpoints include the latest model and optimizer, the best model and loss
history, LR and early-stopping state, the recovery budget, AMP scaler state
when active, and random generators. The CLI also saves the actual train,
validation and test assignments and CV folds, including unseeded splits.
``parrot-optimize`` retains the TPE state, tried configurations, promotion
progress and each fold's latest completed epoch. A checkpoint from final
cross-validated retraining resumes that phase without rerunning the search.
Search stages are checkpointed as well as the requested epoch interval.
The interval counts successful trial/fold epochs across the search.

Resume checks input checksums, data/encoding identities, architecture, batching,
training policies, device type and PyTorch version before continuing. Inputs
must remain available and unchanged. Changing losses, regularization, AMP mode,
LR policy or batch size is a new experiment and is rejected. The output path
and reporting verbosity can change; the CUDA device index can change if the
device supports the same resolved AMP dtype. Exact numerical continuation
requires compatible hardware and software.

For fixed-iteration ``parrot-train`` with fixed, exponential or plateau LR,
``--epochs`` can increase the total training duration. Automatic-stopping
patience, cosine horizons and search budgets must retain their original planned
epochs. Recovery budgets continue across resume. At most the work since the
last checkpoint needs replaying; checkpoints are at epoch boundaries, not in
the middle of a batch. Custom Python loaders require ``num_workers=0`` and
PARROT's ``SequenceDataset`` for checkpoint/resume.

Loader validation includes sampling policies: replacement, sample count, weights,
subset indices, and length-batching settings. Supported samplers are PARROT's
``LengthBatchSampler`` and PyTorch's ``SequentialSampler``, ``RandomSampler``,
``WeightedRandomSampler`` and ``SubsetRandomSampler``, with the standard
``BatchSampler`` where needed. Custom samplers and sampler subclasses are
rejected because their additional state cannot be verified or restored.

The training checkpoint is separate from ``<network>.pt``, the best network
used by ``parrot-predict`` and ``Predictor``. An inference file cannot be passed
to ``--resume``. Interrupted checkpoint writes leave the previous complete file
in place. Checkpoints consume CPU memory and disk space for optimizer and
per-fold states in addition to model weights; searches can have large files.

.. _parrot-amp:

Opt-in CUDA mixed precision
---------------------------

``--amp`` enables automatic mixed precision (AMP). ``--amp auto`` chooses native
BF16 when the selected GPU supports it, otherwise FP16. ``--amp bfloat16`` and
``--amp float16`` select explicitly; ``--amp off`` is the default. This
implementation supports CUDA GPUs with compute capability 7.0 or newer and
requires native BF16 support for that dtype. It checks the selected ``--gpu-id``.
CPU and MPS AMP are not offered, and unsupported requests fail without silently
falling back. PyTorch supports other AMP backends, but PARROT's implementation
currently targets CUDA.

Parameters and optimizer state remain FP32. Autocast wraps the network forward
pass; loss arithmetic, including squared errors and Gaussian NLL, is FP32.
Validation, model-error fitting and prediction keep PARROT's existing FP32
behavior. Every AMP batch checks its loss and unscaled gradient norm before
an optimizer update. Clipping is applied after unscaling, and backward runs
outside autocast. Inputs and model/optimizer tensors are also checked; a
nonfinite post-update state stops or enters the separately enabled rollback.

FP16 uses dynamic loss scaling. If scaled gradients overflow, PARROT lowers the
scale and replays the same retained batch with its original randomness and
buffers, allowing at most eight scaling retries. It never silently discards
the batch. Nonfinite forward losses, nonfinite BF16 gradients and exhausted
scaling retries stop or enter the separately enabled epoch recovery. Scaler
state continues across search stages and is included in disk and rollback
snapshots. BF16 does not use gradient scaling. Continuous logs record the dtype,
scaling retries and loss scale.

AMP can change rounding and may improve speed or memory use depending on the
architecture and GPU; compare matched-seed pilots with FP32 on your hardware.
It does not guarantee stable training or repair an unsuitable learning rate.
See the `PyTorch AMP examples <https://docs.pytorch.org/docs/2.8/notes/amp_examples.html>`_
for the underlying unscaling and clipping rules.

Hyperparameter searches
------------------------

``parrot-optimize`` applies the chosen controls to every trial, fold and stage,
and to final training after cross-validation. These controls are fixed settings,
not searched hyperparameters. The learning rate searched is the initial rate;
cosine uses the full requested training horizon rather than a short first stage.
Schedule, warmup and recovery state continue when a trial is promoted.

The recovery budget persists across a fold's stages, with a separate budget for
each fold and a fresh budget for final retraining. A numerical failure that
cannot be recovered marks that trial as failed, with its reason in
``_hyperparameter_trials.tsv``; the search can continue with other configurations.

.. _parrot-training-diagnostics:

Training records
-----------------

Active settings and a summary of the rates actually used are recorded in
``<network>_info.txt``. Progress output includes the rate used when learning-rate
control or recovery is active. Two optional files supplement the usual outputs:

* ``<network>_training_history.tsv`` is written when a schedule, warmup or
  recovery is enabled. Each successful epoch has one row: ``epoch``,
  ``training_loss``, ``validation_loss``, ``learning_rate`` (the rate used for
  that epoch) and ``recoveries`` (the cumulative count). Gradient checks,
  clipping or equal-protein weighting alone do not create this file, unless
  ``--training-log`` is given.
* ``<network>_recovery_events.json`` is written on successful completion if
  any recoveries occurred. Each event records its epoch, cumulative attempt,
  old and new rate, and failure reason. Events are also printed as they occur,
  unless ``--silent`` is given.

``--training-log`` writes ``<network>_training_history.tsv`` continuously,
including with fixed-rate training. Each successful epoch is flushed to disk
with its losses, rate, recoveries, training/validation times, AMP scaling retries,
dtype and loss scale, and gradient summaries when checks, clipping, recovery
or AMP collect them. It stays available if the run fails. On resume, records
after the last checkpoint are discarded and replayed epochs are recorded once.

For searches, ``--training-log`` writes
``<network>_search_training_history.tsv`` for every successful trial/fold epoch,
including the trial, fold, epoch, losses, rate, recoveries, elapsed seconds and
AMP dtype, scaling retries and loss scale, plus available gradient summaries.
Final CV retraining writes its own
``<network>_training_history.tsv``. The usual schedule history for the winning
network is still exported on successful completion.

The history covers all successful epochs, although the saved network is from
the epoch with the best validation loss. Without ``--training-log``, the usual
schedule history and recovery JSON are written at the end of a successful run.
They are not resumable checkpoints. For
``parrot-optimize``, they describe the winning network's training, or the final
retraining after cross-validation, rather than every discarded trial.
