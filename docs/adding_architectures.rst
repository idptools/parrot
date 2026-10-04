==========================
Adding a new architecture
==========================

PARROT separates *what* a network is from *how* it is trained, evaluated, saved and optimized. Everything except the network itself — reading data, padding batches, masking padded residues out of the loss, training, testing, writing predictions, saving networks and hyperparameter optimization — works for any architecture that follows a small interface. Adding an architecture therefore means writing one module and adding one import.

How the pieces fit together
---------------------------

* ``parrot/tasks.py`` defines ``Task``: whether data is sequence-mapped (``'sequence'``) or residue-mapped (``'residues'``), and the number of classes (1 for regression).
* ``parrot/architectures/`` contains the architectures. ``base.py`` defines the ``ParrotArchitecture`` base class and the ``HyperParameter`` declaration, ``registry.py`` maps names to architectures, ``layers.py`` holds shared building blocks, each neural architecture lives in its own module (``brnn.py``, ``composition.py``, ``cnn.py``, ``unet.py``, ``transformer.py``), and ``classical.py`` holds the simple scikit-learn models.
* ``parrot/encode_sequence.py`` defines the encoders that turn sequences into vectors (one-hot, biophysics, or a user file).
* ``parrot/train_network.py`` trains, evaluates and runs networks; ``parrot/reporting.py`` writes predictions, metrics and figures; ``parrot/checkpoint.py`` saves and loads networks.
* ``parrot/scripts/common.py`` holds everything ``parrot-train`` and ``parrot-optimize`` share, including generating a command-line flag for every architecture hyperparameter.

The interface
-------------

An architecture is a subclass of ``ParrotArchitecture`` with:

* ``name``: the name used with ``--architecture`` and stored in saved networks.
* ``description``: a one-line description.
* ``hyperparameters``: a tuple of ``HyperParameter`` declarations. Each one gets a command-line flag (``hidden_size`` becomes ``--hidden-size``, plus any ``cli_aliases``). A hyperparameter is an ``int``, a ``float``, a ``bool`` (an on/off switch, whose flag turns it off if it is on by default, e.g. ``--attention-pooling-off``, or on otherwise), or a ``str`` with a fixed set of ``choices`` (e.g. a pooling method); numbers can have a ``minimum`` and ``maximum`` and, if ``parrot-optimize`` should tune them, a ``search_range`` (searched on a log scale if ``log_scale=True``); settings with a fixed set of values (choices or on/off) can instead declare ``search_choices``, the values ``parrot-optimize`` tries. Architectures that use the same hyperparameter name share the flag, so give shared hyperparameters (like ``dropout``) the same name and type everywhere; ``layers.py`` declares the shared on/off options (``GLOBAL_CONTEXT``, ``ATTENTION_POOLING``, ``LOGLENGTH_FEEDFORWARD``) for reuse.
* When you add a hyperparameter to an existing architecture, networks saved before it existed do not record it. Give it a ``legacy_value`` (the value that reproduces those networks, e.g. ``False`` for a new feature) and they will be rebuilt with that value rather than the default, so they still load and predict as before.
* ``__init__(self, input_size, task, <one keyword argument per hyperparameter>)``, which must pass the hyperparameter values to ``super().__init__`` so they are saved with the network.
* ``forward(self, x, lengths=None)``. ``x`` is a zero-padded batch of shape ``[batch, padded_length, input_size]`` and ``lengths`` holds the true length of each sequence (``None`` for an unpadded batch). Return ``[batch, num_classes]`` for sequence-mapped tasks or ``[batch, padded_length, num_classes]`` for residue-mapped tasks. Classification outputs are unnormalized logits.

The one rule ``forward`` must follow is that **a sequence's output must not depend on how much padding it was given**. ``parrot/architectures/layers.py`` provides what most architectures need to guarantee this:

* ``resolve_lengths`` and ``padding_mask`` give the true lengths and a mask of real residues;
* ``AttentionPool``, ``GlobalContext`` and ``sequence_features`` provide attention pooling, a whole-sequence summary for convolutional networks, and the pooled (plus optional log-length) vector for sequence tasks;
* re-applying the mask after any layer that can write into padded positions (convolutions, biases, normalization) keeps padding at zero so it cannot leak into real residues;
* ``ChannelLayerNorm`` normalizes each position on its own (batch normalization would mix statistics across padded positions and other sequences, so avoid it);
* ``masked_pool`` averages or max-pools over real residues only, for sequence-mapped tasks.

Outputs at padded positions of residue-mapped tasks are ignored by the loss, so they can be anything.

Two optional class attributes cover models that do not fit the default pattern: ``supported_datatypes`` (default ``('sequence', 'residues')``) restricts an architecture to one kind of data, and ``trained_by_gradient_descent = False`` marks models that are fitted in one step rather than trained epoch by epoch. For the latter, subclass ``ClassicalModel`` in ``parrot/architectures/classical.py``: it already computes sequence features, fits, predicts and saves, so a new scikit-learn model only needs its hyperparameters and a ``make_estimator()`` method returning the estimator.

Example
-------

``parrot/architectures/composition.py`` is the simplest complete example, and ``cnn.py`` shows a convolutional network using the shared layers. A minimal skeleton looks like this (saved as ``parrot/architectures/my_architecture.py``):

.. code-block:: python

    import torch
    import torch.nn as nn

    from parrot.architectures.base import HyperParameter, ParrotArchitecture
    from parrot.architectures.layers import masked_pool, padding_mask, resolve_lengths
    from parrot.architectures.registry import register_architecture


    @register_architecture
    class MyArchitecture(ParrotArchitecture):
        name = 'my_architecture'
        description = 'one-line description shown in --help'
        hyperparameters = (
            HyperParameter('hidden_size', int, default=32, help='width of the hidden layer',
                           minimum=1, search_range=(8, 128)),
            HyperParameter('pooling', str, default='mean', choices=('mean', 'max'),
                           help='sequence tasks only: how residue features are pooled'),
        )

        def __init__(self, input_size, task, hidden_size, pooling):
            super().__init__(input_size, task, {'hidden_size': hidden_size, 'pooling': pooling})
            self.pooling = pooling
            self.hidden = nn.Linear(input_size, hidden_size)
            self.output = nn.Linear(hidden_size, task.num_classes)

        def forward(self, x, lengths=None):
            lengths = resolve_lengths(x, lengths)
            mask = padding_mask(lengths, x.shape[1])
            features = torch.relu(self.hidden(x))

            if self.task.per_residue:
                return self.output(features)
            return self.output(masked_pool(features, mask, self.pooling))

Then import the module in ``parrot/architectures/__init__.py`` next to the built-in architectures so it is registered whenever PARROT is imported:

.. code-block:: python

    from parrot.architectures import brnn, classical, cnn, composition, my_architecture, transformer, unet  # noqa: E402,F401

That is all. The new architecture can now be used everywhere:

.. code-block:: bash

    $ parrot-train data/seq_regress_dataset.tsv my_network.pt -d sequence -c 1 --architecture my_architecture --hidden-size 64
    $ parrot-optimize data/seq_regress_dataset.tsv my_network.pt -d sequence -c 1 --architecture my_architecture
    $ parrot-predict data/seqfile.txt my_network.pt predictions.txt

and from Python with ``parrot.build_network('my_architecture', input_size, task, {...})`` or ``parrot.py_predictor.Predictor('my_network.pt')``.

Testing a new architecture
--------------------------

The test suite reads the list of registered architectures, so a new architecture is automatically checked for the behaviour every architecture must have: output shapes for every task type it supports, padding invariance (with sequences from 1 to 22 residues), gradients flowing through padded batches (or one-step fitting), other encoding sizes, training reducing the loss, predictions, and save/load round trips through ``parrot-predict`` and ``Predictor``. Just run ``pytest``; on a machine with an NVIDIA (CUDA) or Apple-silicon (MPS) GPU, ``tests/test_devices.py`` also checks that the architecture gives the same output on the GPU as on the CPU. Networks may run on the CPU, a CUDA GPU or MPS, so create any tensors inside ``forward`` on ``x.device`` and keep them float32: MPS does not support float64. To also check non-default settings, add them to ``CONFIGURATIONS`` in ``tests/test_architectures.py``, and add tests for anything specific to your architecture there.
