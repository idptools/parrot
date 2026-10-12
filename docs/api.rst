Module Documentation
====================

This page documents PARROT's Python code, module by module. Most users only need the command-line tools, or :class:`~parrot.py_predictor.Predictor` to use a trained network in Python.

Using PARROT from Python
------------------------

To predict with a network trained by ``parrot-train`` or ``parrot-optimize``, load it into a :class:`~parrot.py_predictor.Predictor`. The network file records its architecture, task and encoding, so only its path is needed:

.. code-block:: python

    from parrot import Predictor

    predictor = Predictor('network.pt')    # Predictor('network.pt', device='auto') uses a GPU if there is one
    values = predictor.predict('MKVLAAGIVGLLLA')

``predict`` takes one sequence at a time and returns a NumPy array: for regression, the predicted value (or one value per residue, or the predicted matrix); for classification, the probability of each class (for every residue or matrix cell, for residue and matrix networks).

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
