Module Documentation
====================

This page documents PARROT's Python code, module by module. Most users only need the command-line tools, or :class:`~parrot.py_predictor.Predictor` to use a trained network in Python.

Using PARROT from Python
------------------------

To predict with a network trained by ``parrot-train`` or ``parrot-optimize``, load it into a :class:`~parrot.py_predictor.Predictor`. The network file records its architecture, task and encoding, so only its path is needed:

.. code-block:: python

    from parrot import Predictor

    predictor = Predictor('network.pt')    # on a GPU if there is one; device='cpu' for the CPU
    values = predictor.predict('MKVLAAGIVGLLLA')
    all_values = predictor.predict_batch(['MKVLAAGIVGLLLA', 'MSDNEEGQSKRRQ'])

``predict`` takes one sequence at a time and returns a NumPy array: for regression, the predicted value (or one value per residue, or the predicted matrix); for classification, the probability of each class (for every residue or matrix cell, for residue and matrix networks). ``predict_batch`` takes a list of sequences and returns a list of these arrays, one per sequence in the same order. It predicts the sequences together, as ``parrot-predict`` does (every sequence is encoded first, then the network runs over batches of similar length), so for more than a handful of sequences it is much faster than ``predict`` on each. Its predictions match those of ``predict`` up to float32 rounding, not exactly: which sequences share a batch changes the order of the floating-point operations. For a network trained on protein language model embeddings, the model also runs in batches, and its float32 rounding can move a value of the stored (16-bit) embeddings by one step, so the predictions can differ by more.

Predictions run on a GPU if there is one (CUDA, else Apple-silicon MPS) and otherwise on the CPU, as ``parrot-predict``'s do. ``device='cpu'`` uses the CPU anyway, which can be faster for a few sequences at a time and is needed inside forked worker processes (where MPS cannot be used); on a machine with several NVIDIA GPUs, ``Predictor('network.pt', gpu_id=1)`` runs on the second, as ``parrot-predict --gpu-id 1`` does.

Networks can also be built, trained and saved in Python, with the same functions the command-line tools use. For example, to train a CNN on one of the example datasets in the **/data** folder, split by the example split-file:

.. code-block:: python

    import parrot

    # Sequence-mapped regression (one value per sequence), with one-hot encoding
    task = parrot.Task('sequence', 1)
    encoder = parrot.get_encoder('onehot')
    train_set, val_set, test_set = parrot.split_data('data/seq_regress_dataset.tsv', task, encoder=encoder,
                                                     split_file='data/splitfile.txt')

    # Train a CNN; afterwards the network holds the weights of its best epoch on the validation set
    network = parrot.build_network('cnn', len(encoder), task, {'channels': 32})
    train_loader = parrot.make_dataloader(train_set, task, batch_size=32, shuffle=True)
    val_loader = parrot.make_dataloader(val_set, task, batch_size=32)
    train_losses, val_losses = parrot.train(network, train_loader, val_loader, task, 'cpu', learning_rate=0.001,
                                            n_epochs=20)

    # Save it (parrot-predict and Predictor can then use it), and predict new sequences
    parrot.save_network('network.pt', network, encoder)
    outputs = parrot.predict_sequences(network, ['MKVLAAGIVGLLLA'], encoder, 'cpu')

:func:`~parrot.prediction.predict_sequences` returns the network's raw output for each unique sequence (for classification, unnormalized logits), and :func:`~parrot.prediction.predict_labeled` runs a network over a labelled dataset such as ``test_set``. The architectures and their hyperparameters are described on the :doc:`usage/architectures` page, and :doc:`adding_architectures` explains how to add your own.

Optional training controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default call above uses a fixed learning rate, summed labelled losses and
no gradient clipping or numerical recovery. Import
:class:`~parrot.training_control.TrainingOptions` and
:class:`~parrot.training_control.TrainingControlState` from
``parrot.training_control`` to enable and inspect the optional policies:

.. code-block:: python

    from parrot.training_control import TrainingOptions, TrainingControlState

    options = TrainingOptions(lr_schedule='plateau', lr_patience=3,
                              lr_min=1e-6, recover_nonfinite=2,
                              optimizer='adamw', weight_decay=0.01)
    state = TrainingControlState()
    train_losses, val_losses = parrot.train(
        network, train_loader, val_loader, task, 'cpu', learning_rate=0.001,
        n_epochs=20, check_finite_gradients=True,
        training_options=options, control_state=state,
        checkpoint_path='run_training.pt', training_log='run_training.tsv')
    diagnostics = state.report()  # learning_rate_history and any recovery_events

Here ``network`` and the loaders are built as in the preceding example; use a
fresh network to compare a controlled run with a default run. Clipping is a
separate argument to ``train``: ``gradient_clip_norm=...`` enables it at a
positive, finite threshold and implies finite checks. See
:doc:`usage/training_controls` for calibration, schedule timing and recovery
limits. Recovery requires loaders with ``num_workers=0``.

``TrainingOptions(equal_protein_weighting=True)`` averages labelled losses
within each protein for residue or matrix tasks. When evaluating such a run
outside ``train``, pass ``equal_protein_weighting=True`` to
:func:`~parrot.train_network.evaluate_loss` and
:func:`~parrot.prediction.predict_labeled` too, so the reported loss has the
same meaning. Standard performance metrics still pool labelled values.
Loss weighting is not inferred from the saved inference checkpoint.

For cosine annealing, ``train`` resolves an omitted ``lr_anneal_epochs`` from
``n_epochs`` minus warmup. Give an explicit horizon with
``stop_condition='auto'``. When constructing
:class:`~parrot.hyperparameter_search.TrialTrainer` directly, resolve options
once against the full search horizon with ``options.resolved(n_epochs)`` before
passing them to its shorter stages; the CLI does this automatically. Its state
continues across promotions.

``state.epoch_history`` records successful epochs when learning-rate control or
recovery is active, and ``state.recovery_events`` records rollbacks. The Python
API returns these diagnostics in memory. The explicit ``checkpoint_path`` and
``training_log`` arguments in the example also write a resumable checkpoint and
flushed epoch records. ``checkpoint_every`` defaults to 1. Resume with the same
model, task, data and options, adding ``resume_from='run_training.pt'``; for a
Python-created checkpoint, use this API rather than CLI ``--resume``. Checkpoints
require PARROT ``SequenceDataset`` loaders with ``num_workers=0`` and supported
samplers; their sampling policies must match the checkpoint (see
:ref:`parrot-training-resume`). Custom sampler subclasses are rejected. Use
``on_improvement`` to separately save the best inference network during training.
See :ref:`parrot-training-resume` for validation and allowed duration changes.

Choose a loss with ``Task('residues', 1, regression_loss='huber', huber_delta=0.5)``;
the task is saved with the network, so evaluation uses the same loss. Loss choices
are ``auto`` (the original datatype-dependent defaults), ``mae``, ``mse`` and
``huber``. A selected loss is incompatible with classification or uncertainties.
``TrainingOptions(amp='auto')`` enables CUDA AMP on a compatible GPU, with
``amp='float16'`` or ``'bfloat16'`` to select explicitly. It is unsupported on
CPU and MPS in PARROT and never silently falls back. BRNN training requires
``amp='off'``: ``train``, ``TrialTrainer`` and the lower-level training functions
reject enabled AMP for BRNNs. Saved BRNN AMP configurations cannot resume;
start a new FP32 run. Other neural architectures retain CUDA AMP support.
See :ref:`parrot-amp`.

The parrot package
------------------

.. automodule:: parrot

``import parrot`` gives these names, each documented below in the module that defines it:

* Encoding sequences: :class:`~parrot.sequence_encoder.SequenceEncoder`, :func:`~parrot.encode_sequence.get_encoder`, :func:`~parrot.encode_sequence.encoder_from_config`, :class:`~parrot.encode_sequence.OneHotEncoder`, :class:`~parrot.encode_sequence.BiophysicsEncoder`, :class:`~parrot.encode_sequence.UserEncoder`, :class:`~parrot.encode_sequence.StandardizedEncoder`, :func:`~parrot.encode_sequence.one_hot`, :func:`~parrot.encode_sequence.biophysics`, :func:`~parrot.encode_sequence.parse_encode_file`, and the tables ``ONE_HOT`` and ``BIOPHYSICS`` (in ``encode_sequence.py``).
* Reading datafiles, splitting them into sets and batching them: :func:`~parrot.process_input_data.parse_file`, :func:`~parrot.process_input_data.read_tsv_raw`, :func:`~parrot.process_input_data.load_and_split`, :func:`~parrot.process_input_data.split_data`, :func:`~parrot.process_input_data.split_data_cv`, :func:`~parrot.process_input_data.cross_validation_folds`, :func:`~parrot.process_input_data.read_split_file`, :func:`~parrot.process_input_data.write_split_file`, :class:`~parrot.process_input_data.SequenceDataset` and :func:`~parrot.process_input_data.make_dataloader`.
* The task and the architectures: :class:`~parrot.tasks.Task`, :class:`~parrot.architectures.base.ParrotArchitecture`, :class:`~parrot.architectures.hyperparameters.HyperParameter`, :func:`~parrot.architectures.registry.register_architecture`, :func:`~parrot.architectures.registry.get_architecture`, :func:`~parrot.architectures.registry.available_architectures`, :func:`~parrot.architectures.registry.build_network`, and the original BRNN classes :class:`~parrot.brnn_architecture.BRNN_MtO` and :class:`~parrot.brnn_architecture.BRNN_MtM`.
* Training, saving and predicting: :func:`~parrot.train_network.train`, :func:`~parrot.train_network.evaluate_loss`, :func:`~parrot.checkpoint.save_network`, :func:`~parrot.checkpoint.load_network`, :func:`~parrot.prediction.predict_sequences`, :func:`~parrot.prediction.predict_labeled` and :class:`~parrot.py_predictor.Predictor`.
* The errors PARROT raises for input it cannot use: :class:`~parrot.parrot_exceptions.ParrotException` and :class:`~parrot.parrot_exceptions.IOExceptionParrot`.
* ``__version__``: the version of PARROT.

tasks.py
--------

.. automodule:: parrot.tasks
   :members:

architectures
-------------

.. automodule:: parrot.architectures

.. automodule:: parrot.architectures.base
   :members:

.. automodule:: parrot.architectures.hyperparameters
   :members:

.. automodule:: parrot.architectures.registry
   :members:

.. automodule:: parrot.architectures.layers
   :members:

.. automodule:: parrot.architectures.brnn
   :members:

.. automodule:: parrot.architectures.composition
   :members:

.. automodule:: parrot.architectures.cnn
   :members:

.. automodule:: parrot.architectures.unet
   :members:

.. automodule:: parrot.architectures.transformer
   :members:

.. automodule:: parrot.architectures.classical
   :members:

brnn_architecture.py
--------------------

.. automodule:: parrot.brnn_architecture
   :members:

sequence_encoder.py
-------------------

.. automodule:: parrot.sequence_encoder
   :members:

encode_sequence.py
------------------

.. automodule:: parrot.encode_sequence
   :members:

plm.py
------

.. automodule:: parrot.plm
   :members:

embedding_context.py
--------------------

.. automodule:: parrot.embedding_context
   :members:

nardini.py
----------

.. automodule:: parrot.nardini
   :members:

process_input_data.py
---------------------

.. automodule:: parrot.process_input_data
   :members:

train_network.py
----------------

.. automodule:: parrot.train_network
   :members:

training_control.py
-------------------

.. automodule:: parrot.training_control
   :members:

training_precision.py
---------------------

.. automodule:: parrot.training_precision
   :members:

training_persistence.py
-----------------------

.. automodule:: parrot.training_persistence
   :members:

uncertainties.py
----------------

.. automodule:: parrot.uncertainties
   :members:

prediction.py
-------------

.. automodule:: parrot.prediction
   :members:

checkpoint.py
-------------

.. automodule:: parrot.checkpoint
   :members:

reporting.py
------------

.. automodule:: parrot.reporting
   :members:

metrics.py
----------

.. automodule:: parrot.metrics
   :members:

figures.py
----------

.. automodule:: parrot.figures
   :members:

hyperparameter_search.py
------------------------

.. automodule:: parrot.hyperparameter_search
   :members:

search_space.py
---------------

.. automodule:: parrot.search_space
   :members:

tpe_sampler.py
--------------

.. automodule:: parrot.tpe_sampler
   :members:

tools.splitting and tools.mmseqs_search
---------------------------------------

.. automodule:: parrot.tools.splitting
   :members:

.. automodule:: parrot.tools.mmseqs_search
   :members:

tools.kmer_similarity
---------------------

.. automodule:: parrot.tools.kmer_similarity
   :members:

tools.preproc, tools.class_balancing and tools.dataset_warnings
---------------------------------------------------------------

.. automodule:: parrot.tools.preproc
   :members:

.. automodule:: parrot.tools.class_balancing
   :members:

.. automodule:: parrot.tools.dataset_warnings
   :members:

matrices
--------

.. automodule:: parrot.matrices
   :members:

.. automodule:: parrot.matrices.data
   :members:

.. automodule:: parrot.matrices.base
   :members:

.. automodule:: parrot.matrices.layers
   :members:

.. automodule:: parrot.matrices.output
   :members:

.. automodule:: parrot.matrices.figures
   :members:

.. automodule:: parrot.matrices.architectures.unet2d
   :members:

devices.py
----------

.. automodule:: parrot.devices
   :members:

reproducibility.py
------------------

.. automodule:: parrot.reproducibility
   :members:

parrot_exceptions.py
--------------------

.. automodule:: parrot.parrot_exceptions
   :members:

py_predictor.py
---------------

.. automodule:: parrot.py_predictor
   :members:
