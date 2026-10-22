==============================
 Diagnosing numerical failures
==============================

A nonfinite loss, nonfinite gradient entries and overflow in a norm reduction
are different failures. The :doc:`training_controls` guards distinguish actual
NaN/Inf gradients from finite gradients whose ordinary norm reduction overflows.
The latter can be clipped using a stable norm. Clipping cannot repair entries
that already became NaN/Inf during backward.

Capturing a training failure
----------------------------

Both neural training commands accept ``--capture-nonfinite FILE``. For example:

.. code-block:: bash

    parrot-optimize brnn --input-file data.tsv --split split.txt \
        -d residues --regression --device cuda --amp bfloat16 \
        --output-network optimized.pt --capture-nonfinite failure.pt

This enables finite input, loss, gradient and post-update state checks. The
first numerical failure saves the retained batch, model/optimizer state from
before that batch, random generators, precision/scaler state and runtime
metadata. Only one capture is written; an existing file is never replaced.
The option adds CPU copies before every batch until a failure and should be
used for debugging runs. It does not enable clipping, retries or batch skipping.
Keep the capture separate from the inference network, training checkpoint and
logs. Use ``--checkpoint`` separately for resumable training.

Replay a captured per-residue BRNN regression failure on the training machine:

.. code-block:: bash

    python -m parrot.scripts.diagnose_brnn \
        --replay failure.pt --output-dir failure_replay

The output directory must be new. Replay loads tensors and plain metadata with
``weights_only=True`` and requires the same device type as the capture. It
restores the pre-batch model, optimizer and random state for every comparison:
FP32 and BF16, each with PARROT's unpacked path and PyTorch's packed LSTM, plus
BF16 with the recurrent computation held in FP32. Parameters remain FP32. Replay
retains the capture's loss, weighting, optimizer and clipping setting.
It never updates the production model or overwrites the capture.

Matched precision pilots on real data
-------------------------------------

Use the Python environment and GPU where training failed, with this checkout
installed (for example, ``python -m pip install -e .`` from the repository root).
The report records the imported PARROT source location so that a stale install
can be spotted. Then run:

.. code-block:: bash

    python -m parrot.scripts.diagnose_brnn \
        --input-file data.tsv --split split.txt --output-dir brnn_diagnostic \
        --hidden-size 52 --num-layers 2 --learning-rate 0.001 \
        --batch 128 --epochs 30 --seed 42

Substitute the hyperparameters of a failed search trial. This command is a
fixed-configuration pilot, not a hyperparameter search. The default modes are
``off`` and ``bfloat16``. Each starts from identical weights, with epoch batch
indices generated from an independent, fixed CPU RNG stream; the per-batch
index hashes allow the order to be checked. GPU arithmetic and dropout can
still differ between precisions or implementations. A matching seed does not
guarantee bitwise deterministic CUDA execution. Packed and unpacked dropout
implementations need not draw identical masks when dropout is nonzero.

Training uses the actual PARROT loss/AMP/check/clipping/update functions, summed
per-residue MSE, one-hot encoding and Adam, without recovery. Validation runs
in FP32 on the supplied validation split; the test split is unused. This pilot
does not reproduce other losses, encodings, schedules, recovery or an entire
search history. Capture the original run when those settings matter.

Clipping is off in the diagnostic by default. Repeat in a new output directory
with, for example, ``--gradient-clip-norm 1`` to test that candidate threshold;
calibrate it against norm distributions and validation performance rather than
treating it as a universal value. ``--max-batches N`` caps batches per epoch for
a shorter probe (default 0: all batches); it changes how much data is trained
on, so omit it when reproducing a full epoch. ``--batching length`` is the
default; ``--batching random`` selects the other production batching policy.

The pilot saves the first numerical failure separately for each mode and
automatically replays each against both precisions and implementations. A mode
stops on its first failure; batches are never skipped. BF16 requires native
CUDA support; an unsupported mode is recorded as unavailable and is not
silently emulated or replaced with FP16. ``--modes off float16`` explicitly
tests the FP16 scaler/retry pathway instead.

Synthetic smoke test
---------------------

Without an input file, the diagnostic uses synthetic proteins of lengths 32,
257 and 2,049 with bounded targets:

.. code-block:: bash

    python -m parrot.scripts.diagnose_brnn --output-dir brnn_smoke --epochs 3

``--synthetic-lengths`` changes those lengths. The default device is CUDA;
``--device cpu --modes off`` provides a CPU smoke test. Synthetic validation
reuses the synthetic data and checks numerical behavior only. Passing this
smoke test does not establish stability on a real dataset.

Reading the outputs
--------------------

``environment.json`` records PyTorch, its actual CUDA runtime and cuDNN versions,
GPU model/capability, native BF16 availability, FP32 precision policy and options.
``summary.json`` contains pilot outcomes, elapsed pilot time, validation loss
and replay outcomes. ``fp32_batches.jsonl`` and ``bfloat16_batches.jsonl`` are
flushed after each batch and record loss, the ordinary norm reduction and its
overflow flag, stable pre/post-clipping norms, length range,
label count and batch-order hash. Each ``*_failure.pt`` contains pre-batch state
and batch data; disk usage depends on model/optimizer size and sequence lengths.
Replay-only runs write their comparison results in ``summary.json``.

Interpret failures using both counts and norms:

* Finite entries with a very large stable norm indicate exploding gradients;
  optional clipping may constrain subsequent updates.
* Nonzero NaN/Inf counts indicate actual backward failure. Clipping at the end
  of that backward cannot rescue the batch.
* A failure confined to BF16 in a matched pilot supports investigating reduced
  precision. A successful recurrent-FP32 replay helps narrow that investigation.
* Packed/unpacked replay differences are evidence to investigate on that runtime,
  not proof of a cuDNN defect. Replay uses one state and batch; it does not
  establish the stability of an entire subsequent training run.

Preserve ``summary.json``, ``environment.json`` and the batch logs when reporting
an issue. A failure capture additionally enables direct reproduction. The tool
returns normally when it successfully records a numerical failure; inspect the
reported statuses rather than treating exit code 0 as evidence that training
was stable.
