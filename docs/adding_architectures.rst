==========================
Adding a new architecture
==========================

PARROT separates *what* a network is from *how* it is trained, evaluated, saved and optimized. Everything except the network itself — reading data, padding batches, masking padded residues out of the loss, training, testing, writing predictions, saving networks and hyperparameter optimization — works for any architecture that follows a small interface. Adding an architecture therefore means writing one module and adding one import.

How the pieces fit together
---------------------------

* ``parrot/tasks.py`` defines ``Task``: whether data is sequence-mapped (``'sequence'``), residue-mapped (``'residues'``) or a matrix per sequence (``'matrix'``; see the last section), and the number of classes (1 for regression).
* ``parrot/architectures/`` contains the architectures. ``base.py`` defines the ``ParrotArchitecture`` base class (including the readout most architectures end with), ``hyperparameters.py`` defines the ``HyperParameter`` declaration and the hyperparameters several architectures share, ``registry.py`` maps names to architectures and builds networks, ``layers.py`` holds shared building blocks, each neural architecture lives in its own module (``brnn.py``, ``composition.py``, ``cnn.py``, ``unet.py``, ``transformer.py``), and ``classical.py`` holds the simple scikit-learn models.
* ``parrot/sequence_encoder.py`` defines ``SequenceEncoder``, the interface of the encoders that turn sequences into one vector per residue: ``parrot/encode_sequence.py`` holds the per-amino-acid tables (one-hot, biophysics, or a user file), ``parrot/plm.py`` the protein language models and ``parrot/nardini.py`` the NARDINI+ features. An architecture only sees the length of the vectors (``input_size``), so it works with every encoding.
* ``parrot/process_input_data.py`` reads datafiles, splits them into sets and pads sequences into batches.
* ``parrot/train_network.py`` trains networks (and fits the classical models); ``parrot/prediction.py`` runs trained networks; ``parrot/metrics.py``, ``parrot/figures.py`` and ``parrot/reporting.py`` measure, plot and write how a network performs; ``parrot/checkpoint.py`` saves and loads networks.
* ``parrot/hyperparameter_search.py`` (with ``search_space.py`` and ``tpe_sampler.py``) is ``parrot-optimize``'s search, which builds its search space from the architecture's hyperparameter declarations.
* ``parrot/scripts/training_parser.py`` builds the command line of ``parrot-train`` and ``parrot-optimize``: a subcommand for every registered architecture, with a flag for each of its hyperparameters. ``parrot/scripts/common.py`` holds the rest of what the command-line tools share.

The interface
-------------

An architecture is a subclass of ``ParrotArchitecture`` with:

* ``name``: the name of the architecture's ``parrot-train`` and ``parrot-optimize`` subcommand, also stored in saved networks. It must not be the name of another architecture.
* ``description``: a one-line description, shown in the list of architectures in ``parrot-train -h``.
* ``hyperparameters``: a tuple of ``HyperParameter`` declarations. Each one gets a command-line flag (``hidden_size`` becomes ``--hidden-size``, plus any ``cli_aliases``). A hyperparameter is an ``int``, a ``float``, a ``bool`` (an on/off switch, whose flag turns it off if it is on by default, e.g. ``--attention-pooling-off``, or on otherwise), or a ``str`` with a fixed set of ``choices`` (e.g. a pooling method); numbers can have a ``minimum`` and ``maximum`` (inclusive, or exclusive with ``exclusive_minimum=True`` or ``exclusive_maximum=True``) and, if ``parrot-optimize`` should tune them, a ``search_range`` (searched on a log scale if ``log_scale=True``); settings with a fixed set of values (choices or on/off) can instead declare ``search_choices``, the values ``parrot-optimize`` tries. Each architecture's subcommand has its own flags, so a hyperparameter's declaration only has to suit its own architecture. Where several architectures have the same setting, ``hyperparameters.py`` declares it once (``DROPOUT``, ``POOLING``, ``GLOBAL_CONTEXT``, ``ATTENTION_POOLING``, ``LOGLENGTH_FEEDFORWARD``), so use those where they fit, so that the same flag means the same thing everywhere.
* When you add a hyperparameter to an existing architecture, networks saved before it existed do not record it. Give it a ``legacy_value`` (the value that reproduces those networks, e.g. ``False`` for a new feature) and they will be rebuilt with that value rather than the default, so they still load and predict as before.
* ``__init__(self, input_size, task, <one keyword argument per hyperparameter>)``, which must pass every hyperparameter value to ``super().__init__``. The values are checked there (a missing, misspelt or invalid one is an error) and saved with the network, which is how it is rebuilt when loaded.
* ``forward(self, x, lengths=None)``. ``x`` is a zero-padded batch of shape ``[batch, padded_length, input_size]`` and ``lengths`` holds the true length of each sequence (``None`` for an unpadded batch). Return ``[batch, num_classes]`` for sequence-mapped tasks or ``[batch, padded_length, num_classes]`` for residue-mapped tasks. Classification outputs are unnormalized logits.

The one rule ``forward`` must follow is that **a sequence's output must not depend on how much padding it was given**. ``parrot/architectures/layers.py`` provides what most architectures need to guarantee this:

* ``resolve_lengths`` and ``padding_mask`` give the true lengths and a boolean mask of real residues; ``zero_padding`` re-zeroes the padding of a ``[batch, length, features]`` tensor, and ``channel_mask`` turns the mask into the float ``[batch, 1, length]`` form that multiplies a channels-first (convolutional) tensor;
* ``AttentionPool`` and ``GlobalContext`` provide attention pooling and a whole-sequence summary for convolutional networks;
* re-applying the mask after any layer that can write into padded positions (convolutions, biases, normalization) keeps padding at zero so it cannot leak into real residues;
* ``ChannelLayerNorm`` normalizes each position on its own (batch normalization would mix statistics across padded positions and other sequences, so avoid it);
* ``masked_pool`` averages or max-pools over real residues only, for sequence-mapped tasks.

Outputs at padded positions of residue-mapped tasks are ignored by the loss, so they can be anything.

Most architectures end the same way: a linear layer maps each residue's features to its outputs, after (for sequence-mapped tasks) pooling the residues into one vector per sequence by attention or by ``--pooling``, optionally with log(sequence length) added. ``ParrotArchitecture`` provides this readout: call ``self.build_readout(num_features, attention_pooling=..., pooling=..., log_length=...)`` at the end of ``__init__`` and return ``self.readout(features, mask, lengths)`` from ``forward``. It creates the submodules ``attention_pool`` and ``fc``, and the shared ``ATTENTION_POOLING``, ``POOLING`` and ``LOGLENGTH_FEEDFORWARD`` hyperparameters switch its parts on and off.

The names of a network's submodules are stored in saved networks (its weights are saved under names such as ``blocks.0.conv.weight``), so once networks have been saved with an architecture, its submodules must never be renamed.

Two optional class attributes cover models that do not fit the default pattern: ``supported_datatypes`` (default ``('sequence', 'residues')``) restricts an architecture to one kind of data, and ``trained_by_gradient_descent = False`` marks models that are fitted in one step rather than trained epoch by epoch. For the latter, subclass ``ClassicalModel`` in ``parrot/architectures/classical.py``: it already computes sequence features, fits, predicts and saves, so a new scikit-learn model only needs its hyperparameters (including ``FEATURES_HYPERPARAMETER``, the choice of sequence features) and a ``make_estimator(num_samples, num_features)`` method returning the unfitted estimator. These two attributes also decide which options the architecture's subcommand offers: ``-d`` lists only the datatypes it supports (and is not needed if there is only one), only architectures trained by gradient descent get the training options (``-e``, ``-b``, ``-lr`` and so on), and only architectures for matrix data get the matrix-data options.

A third, ``supports_uncertainties`` (default ``True``), says whether the architecture can be trained on regression values with uncertainties (``--uncertainties``; see "Values with uncertainties"). A network trained by gradient descent needs nothing extra for this: the loss and the model error are handled outside the architecture, and the model error is kept in a buffer the base class creates. A model fitted in one step must weight its training values (``ClassicalModel.fit`` passes ``sample_weights()`` to its scikit-learn estimator), so set it to ``False`` for an estimator that cannot take sample weights, as ``knn`` does; its subcommand then does not offer ``--uncertainties``.

Two optional class methods tell ``parrot-optimize`` what not to search. ``unused_hyperparameters(task, input_size)`` returns the hyperparameters that have no effect for a task and encoding, each with the reason, which the search reports and leaves out (e.g. ``composition``'s window for sequence data). ``inactive_hyperparameters(values)`` returns those that have no effect given the values of the others (e.g. the SVM's ``relative_gamma`` with a linear kernel); the search gives them their default, so configurations that differ only in them are tried once. By default both are empty.

Example
-------

``parrot/architectures/cnn.py`` is the best template: a convolutional network using the shared hyperparameters, layers and readout. ``composition.py`` is shorter but builds its own output layers. A minimal skeleton looks like this (saved as ``parrot/architectures/my_architecture.py``):

.. code-block:: python

    import torch
    import torch.nn as nn

    from parrot.architectures.base import ParrotArchitecture
    from parrot.architectures.hyperparameters import (ATTENTION_POOLING, LOGLENGTH_FEEDFORWARD, POOLING,
                                                      HyperParameter)
    from parrot.architectures.layers import padding_mask, resolve_lengths
    from parrot.architectures.registry import register_architecture


    @register_architecture
    class MyArchitecture(ParrotArchitecture):
        """Each residue goes through one hidden layer, then the shared readout."""

        name = 'my_architecture'
        description = 'one-line description shown in --help'
        hyperparameters = (
            HyperParameter('hidden_size', int, default=32, help='width of the hidden layer',
                           minimum=1, search_range=(8, 128)),
            POOLING,
            ATTENTION_POOLING,
            LOGLENGTH_FEEDFORWARD,
        )

        def __init__(self, input_size, task, hidden_size, pooling, attention_pooling, loglength_feedforward):
            super().__init__(input_size, task, {'hidden_size': hidden_size, 'pooling': pooling,
                                                'attention_pooling': attention_pooling,
                                                'loglength_feedforward': loglength_feedforward})
            self.hidden = nn.Linear(input_size, hidden_size)
            self.build_readout(hidden_size, attention_pooling=attention_pooling, pooling=pooling,
                               log_length=loglength_feedforward)

        def forward(self, x, lengths=None):
            lengths = resolve_lengths(x, lengths)
            mask = padding_mask(lengths, x.shape[1])
            features = torch.relu(self.hidden(x))
            return self.readout(features, mask, lengths)

Then import the module in ``parrot/architectures/__init__.py`` next to the built-in architectures so it is registered whenever PARROT is imported:

.. code-block:: python

    from parrot.architectures import brnn, classical, cnn, composition, my_architecture, transformer, unet  # noqa: E402,F401

That is all. The new architecture can now be used everywhere:

.. code-block:: bash

    $ parrot-train my_architecture --input-file data/seq_regress_dataset.tsv --output-network my_network --split random -d sequence --regression --hidden-size 64
    $ parrot-optimize my_architecture --input-file data/seq_regress_dataset.tsv --output-network my_network --split random -d sequence --regression
    $ parrot-predict data/seqfile.txt my_network.pt predictions.txt

and from Python with ``parrot.build_network('my_architecture', input_size, task, {...})`` or ``parrot.Predictor('my_network.pt')``.

(An architecture in a module of your own, outside PARROT, is registered when that module is imported, so it can be used from Python once you have imported it, including to load networks saved with it. The command-line tools only know the architectures PARROT itself imports.)

Testing a new architecture
--------------------------

The test suite reads the list of registered architectures, so a new architecture is automatically checked for the behaviour every architecture must have: output shapes for every task type it supports, padding invariance (with sequences from 1 to 22 residues), gradients flowing through padded batches (or one-step fitting), other encoding sizes, training reducing the loss, predictions, saving and loading (as ``parrot-predict`` loads networks) and ``Predictor``, and its ``parrot-train`` and ``parrot-optimize`` subcommands, which must offer only its own hyperparameter flags. Just run ``pytest``; on a machine with an NVIDIA (CUDA) or Apple-silicon (MPS) GPU, ``tests/test_devices.py`` also checks that the architecture gives the same output on the GPU as on the CPU. Networks may run on the CPU, a CUDA GPU or MPS, so create any tensors inside ``forward`` on ``x.device`` and keep them float32: MPS does not support float64. To also check non-default settings, add them to ``CONFIGURATIONS`` in ``tests/test_architectures.py``, and add tests for anything specific to your architecture there.

``tests/test_snapshots.py`` pins every built-in architecture's parameter names and outputs (under every combination of its on/off options), so that refactoring can never silently break saved networks. A new architecture adds new cases, so that test will ask you to regenerate the reference files once, with ``python -m tests.test_snapshots``; from then on it protects your architecture too.

Architectures for matrix data
-----------------------------

Matrix data (the ``'matrix'`` datatype) gives each sequence a matrix of values with one row per residue: a square map of every pair of residues (``Task.width`` is ``None``), or a fixed number of related, ordered columns per residue (``Task.width`` is that number). Everything specific to it lives in ``parrot/matrices/``, and matrix architectures live in ``parrot/matrices/architectures/``, one module each (``unet2d.py`` is the template).

A matrix architecture subclasses ``MatrixArchitecture`` (``parrot/matrices/base.py``) instead of ``ParrotArchitecture``. It declares ``PAIR_CHANNELS`` and ``PAIR_FEATURES`` among its hyperparameters, calls ``self.build_input_grid(pair_channels, pair_features)`` and ``self.build_matrix_readout(num_features)`` in ``__init__``, and in ``forward``:

* gets a channels-first input grid ``[batch, channels, padded_length, width]``, and the boolean ``[batch, padded_length, width]`` mask of its real cells, from ``self.input_grid(x, lengths)`` (with ``lengths = resolve_lengths(x, lengths)`` first, as in the 1D networks). The grid is built from the per-residue encoding: pair features of residues i and j for square maps, or residue i's features with an embedding of column j;
* runs its own 2D body on it, keeping padded cells at zero (``cell_mask``, ``grid_channel_mask`` and ``ChannelLayerNorm2d`` in ``parrot/matrices/layers.py`` help, as in the 1D networks);
* returns ``self.matrix_readout(features)``: ``[batch, padded_length, width, num_classes]``, made symmetric for symmetric tasks.

Import the module in ``parrot/matrices/architectures/__init__.py`` to register it. ``tests/test_matrix_architectures.py`` reads the registry, so a new matrix architecture is checked automatically for output shapes, padding invariance, symmetric maps, gradients, saving and loading, training, and (with a GPU) the same output on the GPU as on the CPU.
