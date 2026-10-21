# Change log

## PARROT2 (2.0.0.dev0; in development on the `parrot2` branch)

### Requirements

- Python ≥ 3.11, NumPy ≥ 2.3, PyTorch ≥ 2.8.
- scikit-learn ≥ 1.4.
- h5py ≥ 3.10, for the embedding cache of protein language model and NARDINI+ encodings.
- tqdm, for progress bars.
- pandas and seaborn no longer needed.

### Command line

- `parrot-train` and `parrot-optimize` take the architecture as a subcommand, e.g. `parrot-train cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --regression`.
- Each subcommand offers only the options that apply to it: its own hyperparameters, training options only for networks trained epoch by epoch, matrix options only for `unet2d`, `-d` only where there is a choice of datatype.
- Every input named; no positional arguments.
- `--input-file` and `--output-network` (`.pt` added if missing).
- `--regression`, or `--classification --num-classes N`, replace `-c`.
- `--device cpu` replaces `--force-cpu`.
- Commands in PARROT 1 form stop with a message saying what to use instead.
- `--split` required (see Splitting).
- `--seed` in `parrot-train`, `parrot-optimize`, `parrot-cvsplit` and `parrot-preprocess`.
- Every command starts with a header: PARROT's version and the time.
- `--show-examples` on every command: a guide with what it does, worked examples and how it works.
- Help of every command revised for accuracy and consistency.
- Mistakes in the input reported as a one-line message, not a traceback (`PARROT_TRACEBACK=1` shows the traceback too).
- Error messages name the file or option at fault and how to fix it (e.g. `--num-classes` for out-of-range class labels).
- Datafile errors name the data line and what is wrong with it.
- `--exclude-seq-id` error messages count the file's own columns.
- `--datatype`, `--mode` or `--stop` values outside the allowed choices stop with a usage message.
- `parrot-train` and `parrot-optimize` report how the data were split, number epochs from 1 (always showing the last), say which epoch's network was kept, and end with the test-set metrics and the files written.

### Architectures

- New architectures added as a single module (docs: "Adding a new architecture").
- `cnn`: dilated residual convolutional network.
- `unet`: 1D U-Net, adapted from the 2D U-Net on the parrot-lightning branch.
- `transformer`: transformer encoder with rotary or sinusoidal positions.
- `composition`: order-independent baseline.
- `unet2d`: 2D U-Net for matrix data, after Nick's on the parrot-lightning branch.
- Simple models for sequence-mapped data, fitted in one step to composition, length and optionally dipeptide-composition features: `linear`, `svm`, `random_forest`, `knn`.
- `gbrt`: gradient-boosted regression trees (scikit-learn's histogram-based implementation), a further simple model; supports `--uncertainties` and `--probabilistic-classification`.
- CNN and U-Net add a whole-sequence summary to every position by default (`--global-context-off`).
- BRNN, CNN, U-Net and transformer pool residues with learned attention by default (`--attention-pooling-off`); the BRNN previously used its final hidden states.
- BRNN, CNN, U-Net and transformer give their output layer log(sequence length) on sequence tasks by default (`--loglength-feedforward-off`).
- Networks saved before these options existed load with them off.
- BRNN `--dropout` (off by default), on the LSTM output and between stacked LSTM layers; every network trained by gradient descent now has it.
- BRNN dropout not searched by `parrot-optimize`, so BRNN searches try the same configurations as before.
- CNN and U-Net whole-sequence summary no longer draws random numbers when created: a network built with it from a given seed starts with the same other weights as one built without it, as documented (architecture snapshots updated).
- Transformer `--head-size` must be ≥ 2 (with 1, rotary positions did nothing).
- `BRNN_MtO` and `BRNN_MtM` unchanged.

### Training

- Zero-padding masked: the LSTM sees only each sequence's real residues, and padded residues no longer contribute to the loss.
- Predictions no longer depend on how sequences are batched, beyond float32 rounding.
- Previously trained networks load and predict as before, up to float32 rounding.
- Apple-silicon GPUs (MPS) as well as NVIDIA ones: `--device auto|cpu|cuda|mps`, `--gpu-id` to choose among several NVIDIA GPUs.
- Length batching (`--batching length`, the default) jitters each length before sorting: a few percent when the training set is sorted as one pool (most PARROT datasets), less the more pools it fills. The same sequences no longer share a batch every epoch.
- With a given `--seed`, training therefore gives slightly different results than before (training snapshots updated).
- `--stop auto` stops once `-e` epochs in a row have not lowered the validation loss by ≥ 0.5% (relative to the last epoch that did). It stopped one epoch later than documented, and any new best, however small, restarted the count, so a slow steady decline never stopped it.
- Reported test loss uses the same loss function as training and validation.
- Test set predicted in batches of `--batch` (was always 32, so a network that only fitted in memory at a smaller batch could fail after it was saved).
- Full float32 precision on NVIDIA GPUs uses PyTorch's newer `fp32_precision` settings where they exist (PyTorch ≥ 2.9 raises if a process mixes them with the older `allow_tf32` flags; lowering the matmul precision with `torch.set_float32_matmul_precision` broke PARROT's).
- Hyperparameters must be finite numbers (`inf` and `True` refused).
- `--dropout` must be below 1.
- `parrot-train` notes a hyperparameter given that it knows has no effect (e.g. `--window-radius` for sequence data), and `_info.txt` marks it. Not every case caught yet: the pooling options and `--loglength-feedforward-off` do nothing for residue data, but are not noted.
- `--output-network` naming a directory, or `--embedding-cache` with an encoding that has no cache, refused before any data are split or networks trained.
- Few-batches warning fires below 10 batches per epoch, as documented.
- Fixed a crash testing residue-classification networks with NumPy ≥ 1.24.

### Training controls

- All opt-in, for networks trained by gradient descent (`parrot-train` and `parrot-optimize`); without them, training is as before (docs: "Training controls").
- `--lr-schedule cosine|exponential|plateau` (default `fixed`), with `--lr-min`, `--lr-anneal-epochs` (cosine), `--lr-gamma` (exponential), and `--lr-factor`, `--lr-patience`, `--lr-threshold`, `--lr-cooldown` (plateau).
- `--lr-warmup-epochs`: linear warmup to the initial learning rate.
- `--gradient-clip-norm`: global L2 gradient clipping; also turns on finite checks.
- `--check-finite-gradients`: stops on a NaN or infinite loss or gradient norm, before the update.
- `--recover-nonfinite [N]`: on a numerical failure, back to the last good epoch, learning rate cut by `--recovery-lr-factor` (default 0.5), epoch replayed; up to N times (2 if no N given).
- `--regression-loss mae|mse|huber` (`--huber-delta`, default 1); defaults unchanged (MAE for sequences, MSE for residues and matrices); not with `--uncertainties`.
- `--optimizer adamw`; `--weight-decay` for either optimizer (default 0).
- `--equal-protein-weighting` (residue and matrix tasks): labelled losses averaged within each protein, then across proteins, for training, validation and test loss; reported metrics still pool every labelled value.
- `--amp [auto|float16|bfloat16]`: CUDA mixed precision in the training forward pass (`auto`: bfloat16 where supported, else float16); evaluation and prediction stay float32.
- Active controls and a summary of the learning rates used recorded in `_info.txt`.

### Checkpoints and resume

- `--checkpoint [file]` (`parrot-train`, `parrot-optimize`): full training state saved every `--checkpoint-every` epochs (default 1), by default to `<network>_training_checkpoint.pt`.
- `--resume file`: continues a checkpointed run with its saved arguments, data split, optimizer and random-number state; `parrot-train --resume file` alone is enough.
- `parrot-optimize --resume`: the whole search, including TPE state, partly trained folds, promotions and final cross-validated retraining.
- `--training-log`: each epoch's losses, learning rate and timings written as training runs, so kept if the run fails (`<network>_training_history.tsv`; `parrot-optimize` also writes `<network>_search_training_history.tsv` for every trial and fold).
- Checkpoints and saved networks written atomically: a run killed while saving keeps the previous file intact.

### Hyperparameter search: `parrot-optimize`

- Rewritten without GPy/GPyOpt.
- Successive halving (configurations dropped after a few epochs unless among the best) combined with model-based sampling (TPE).
- About 15-20 times cheaper than before with its defaults.
- Configurations scored on the validation set by default; `--cv-folds` for cross-validation.
- Searches settings with fixed choices (e.g. kernel widths) as well as numbers.
- Every trial recorded in `<output>_hyperparameter_trials.tsv`.
- `--trials` replaces `--max-iter` (which still works).
- Architecture hyperparameter flags fix those hyperparameters instead of searching them.
- First stage exactly `--min-epochs` (by default a ninth of `--epochs`): three stages for any `--epochs` from 9 up (e.g. 6, 17 and 50 for `-e 50`, which used to get two).
- A search in which every configuration diverges stops with an error, rather than saving a diverged network.
- Hyperparameters with no effect on the data not searched (e.g. the composition window for sequence data, dipeptide features with protein language model embeddings).
- Configurations that differ only in a setting the rest of the configuration ignores not searched.
- `unet2d` search no longer reaches networks of over a billion parameters (at most about 41 million).
- No longer stops with "Every configuration failed" when the configuration it promoted fails at a later stage (e.g. out of GPU memory) while others trained fine; trains the best of those again instead.
- Simple models: validation loss reported in `_info.txt`, and the search no longer called successive halving.
- `gbrt` search: number of trees, shrinkage, number of leaves, minimum leaf size, L2 penalty, fraction of features considered per split.
- Training controls fixed for every trial, fold and stage; only the initial learning rate searched. Schedules span the full `--epochs` and carry on when a configuration is promoted.
- A trial that fails a numerical check and cannot recover recorded as failed in `_hyperparameter_trials.tsv`; the search continues.

### Data

- Missing residue labels: `nan` in a residue-mapped datafile.
- Matrix data: a matrix of values per sequence, one row per residue. Either a square map of every pair of residues (optionally `--symmetric`) or `--matrix-width N` related values per residue.
- Matrices read from a NumPy `.npz` file (`--matrix-file`), missing cells as NaN; regression or classification.
- `--max-matrix-length` (default 512) refuses sequences whose square maps would not fit in memory.
- Predicted matrices written as `.npz` files by `parrot-train`, `parrot-optimize` and `parrot-predict`; heatmaps of example test sequences in the figures.
- A `.npy` file given as `--matrix-file`, matrices of complex numbers and matrix files holding arrays of Python objects refused with a message.

### Values with uncertainties

- `--uncertainties` in `parrot-train`, `parrot-optimize` and `parrot-preprocess`: regression values with the standard deviation of their measurement error, written `value,SD`.
- Training minimises the Gaussian negative log-likelihood with variance SD² + s².
- Model error s refitted by maximum likelihood to the validation residuals after every epoch.
- Simple models refitted with weights until s settles; `knn` cannot weight its training sequences.
- `--stop auto` then asks for a fall of about 0.005 per labelled value.
- Test statistics add the reduced chi-squared, the fraction of values within 1 and 2 total SDs, the noise ceiling and s.
- `parrot-preprocess --deduplicate merge` combines copies by their inverse-variance weighted mean.

### Data preparation: `parrot-preprocess`

- Now a complete data-preparation tool, with a report of every step.
- Sequence cleaning: `--uppercase`, `--remove-nonstandard`, `--min-length`, `--max-length`.
- Deduplication: `--deduplicate merge|first|drop`.
- Splitting into training, validation and test sets, with a matching split-file.
- `--split homology`: MMseqs2; no set contains a homologue of another.
- `--split kmer`: the same protocol with PARROT's own k-mer search, for when MMseqs2 is not installed; finds closer relatives only.
- `--split random`.
- `--groups`: records in the same group (e.g. overlapping tiles of one protein) always put in the same set.
- Class balancing of the training set only.
- Class balancing removes the fewest sequences needed for the best balance (sequence data), never makes the balance worse (residue data), and keeps the input order.
- `--cb-shuffle-off` no longer has any effect.
- Cleans, deduplicates and splits matrix datafiles.
- `--encode` computes a protein language model or NARDINI encoding of every output sequence ahead of time, into the embedding cache that `parrot-train`, `parrot-optimize` and `parrot-predict` read with `--embedding-cache`: encode once, train any number of networks.
- `--remove-line-duplicates` keeps every line in its original order (used to regroup lines that share an ID).
- Keeps a `--groups` link when deduplication removes one of a group's records.
- Merging duplicates: class labels that are the same number (`1` and `1.0`) are one class.
- Merging duplicates: residue sequences left with no label removed.
- `--deduplicate merge`: copies of a classified sequence whose label is missing (`nan`) no longer count as disagreeing with the others.
- Stops with a message when cleaning or deduplication leaves no lines.
- Spaces around `--delimiter` fields ignored; fields containing whitespace refused (the whitespace-separated output could not hold them).
- Reads datafiles that start with a byte-order mark.
- Checks `--threads`.
- Says so when `--class-balance` finds every label missing.
- No longer crashes on an infinite class label.
- Out-of-range class-balancing options refused before any data are split.
- Report records the random seed (or that none was set), cluster sizes and the initial assignment to sets.
- Report saved when a run fails, ending `FAILED` with the error; successful runs end `COMPLETED` (`PREPROCESSED` while `--encode` is still running).

### Splitting

- `--split` required in `parrot-train` and `parrot-optimize`: a split-file, or one of `parrot-preprocess`'s splitting methods.
- `mmseqs` or `kmer`: similar sequences (found by MMseqs2, or by PARROT's own k-mer search) kept in the same set, so no test sequence has a close relative in the training set.
- `random`: closest to the old default, but keeps identical sequences in the same set.
- The methods aim for `--set-fractions`.
- A file with the name given always read first (a split-file called `random` is still read as a file).
- `kmer` scores a run of shared k-mers by the residues they cover.
- `kmer` E-values calibrated on random sequences. They had been up to tens of times too significant: the overlapping k-mers of one shared stretch counted as independent matches.
- `kmer`: a single short shared word no longer links unrelated sequences or is reported as near-identical. Links slightly fewer distant relatives (in the yeast proteome, 76% of the pairs MMseqs2 links at 40-50% identity, rather than 82%).
- `kmer` no longer understates the chance of a run that covers all of a short sequence (by the chance that a run continues, about 1% for typical compositions).
- `--homology-evalue` above 0.01: the separation check searches at that E-value, so finds every pair that meets the criterion.
- MMseqs2 coverage counts the shorter sequence's residues between the alignment's ends, not alignment columns (which included gaps).
- MMseqs2 always reads the sequences as proteins.
- Similar pairs counted once in the report.
- Separation check no longer removes training records similar only to a validation record that is itself removed.
- `parrot-preprocess --max-separation-rounds` (default 5): removal rounds the separation check may use.
- Separation check: up to that many removal rounds, then a final check. Was five checks in all, so at most four removal rounds; `--split mmseqs` and `kmer` in `parrot-train` and `parrot-optimize` also get the extra round.
- Separation report: for each check, the conflicts between each pair of sets, the identity, E-value and coverage ranges of the matches, and up to three example pairs by ID; for each round, the records and groups removed and how much of each set remains.
- Separation failure: report gives the conflicts left and what another round would remove, and suggests rerunning with more rounds and a fixed `--seed`.
- Split-files must have exactly three lines (blank lines after them aside); a fourth line or a blank first line is an error, rather than misread.
- Split-file values that are not line numbers reported with the file and line.
- Split-file values that only look like line numbers (such as `--2` or `1_0`) refused with the file and line, rather than crashing or being misread.

### Cross-validation

- `parrot-optimize --cv-folds` makes its folds as `--split` makes the sets: identical sequences always in the same fold, and with `--split kmer` or `mmseqs`, clusters of similar sequences too.
- Each fold scored only on held-out sequences not similar to its training sequences (folds used to be drawn at random, so held-out sequences could be copies of training ones).
- `parrot-cvsplit` makes its folds with the same code as `parrot-train`; a given `--seed` gives different (equally valid) split-files than before.
- `parrot-cvsplit` keeps identical sequences in the same set.
- `parrot-cvsplit` keeps every dot of its output prefix (`cv/splits_t0.8` and `cv/splits_t0.9` used to write the same files).
- `parrot-cvsplit --exclude-seq-id` for datafiles without sequence IDs (their first value was taken for the sequence, so identical sequences could be split between sets).
- `parrot-cvsplit` stops with a message if a line's sequence is a number.

### Encodings

- Protein language model embeddings: ESM-2 (`--encode esm2_t12_35M_UR50D` etc.; `plm` extra) and ESM C (`--encode esmc_300m` or `esmc_600m`; `esmc` extra), computed once and cached (`--embedding-cache`).
- NARDINI+ features for sequence-mapped data: `--encode nardini`, or `onehot+nardini` to follow each residue's own encoding.
- NARDINI+ features computed by SPARROW once per sequence, in parallel, into the embedding cache.
- Encoding settings saved with the network, so new sequences are encoded identically.
- `--standardize-encoding`: every feature of the encoding to mean 0 and SD 1 across the characters it defines.
- `--embedding-context` and `--embedding-context-map` (`parrot-preprocess`, `parrot-train`, `parrot-optimize`, `parrot-predict`): each sequence embedded within the full-length protein it was cut from, given a FASTA file of the proteins and a map from each sequence ID to its protein.
- Fragments' embeddings cut out of the proteins' embeddings and cached in their own file.
- Table of where each sequence was embedded written next to the output.
- Networks trained this way record it, so `parrot-predict` and `Predictor` embed new sequences the same way (or on their own, with `--embed-in-isolation`).
- `--embed-mean-subtraction` (`parrot-preprocess`, `parrot-train`, `parrot-optimize`; recorded in the network for `parrot-predict`): each embedding's mean over its residues (in context, the full-length protein's) subtracted before it is stored.
- `--embedding-layer` (`parrot-preprocess`, `parrot-train`, `parrot-optimize`; recorded in the network for `parrot-predict`): embeddings from an earlier layer of the model, numbered from 1; negative numbers count back from the final layer (the default).
- Earlier layers: model run only up to that layer; output without the final layer norm; own cache files (`<model>.layerN.h5`).
- Embeddings too large to store as float16 refused, rather than stored as infinities.
- Several runs can share one `--embedding-cache`: a run reading the cache no longer stops another adding to it (HDF5's file lock gave a raw locking error), and a run that wants to write waits for one that is writing.
- NARDINI features cached per SPARROW version; features from an older SPARROW not reused after an upgrade.
- Progress bar (tqdm) while embedding and computing NARDINI+ features, redrawn every 30 s when the output goes to a file (e.g. a `nohup` log).
- Encoding files: each residue a single character followed by at least one value; may start with a byte-order mark.

### Output and figures

- `<network>_info.txt` for every network, in enough detail for a paper's methods section.
- `_info.txt` contents: the exact command and software versions, the data with its checksum, the split, the encoding, the architecture with its number of parameters, the loss, optimizer and epochs, the search (`parrot-optimize`) and the results.
- `_info.txt` records the training time, time per epoch, training speed and prediction speed (sequences per second).
- Figures drawn with matplotlib alone, in a plain style: Arial text that stays editable in vector editors, solid lines, no titles or grid lines.
- `--include-figs`, regression: density plot of predicted against true test values alongside the scatterplot, both labelled with Pearson's r.
- Scatterplots of more than 10,000 values show 10,000 spread evenly over the test set, with a warning (more would merge into a solid block).
- `--include-figs`, residue networks: `_res_profiles.png`, the true and predicted values along six test sequences spanning the range of accuracy, each labelled with its Pearson's r and mean absolute error (or, for classification, its accuracy).
- `--probabilistic-classification` for residue classification: each residue's class probabilities written as one comma-separated token (e.g. `0.9100,0.0600,0.0300`), so tokens still line up with residues.
- Residue classification: ROC and precision-recall curves and their areas pool every labelled residue.
- A test set with fewer than two labelled values, a constant prediction or a diverged network gives NaN correlations, rather than a crash after the network is saved.
- Class probabilities that are not finite give NaN curve areas and no ROC or precision-recall figures, with a warning.
- Two classes: the area under the ROC curve in the statistics is class 1's, the one the ROC figure shows (was the micro-average over both classes).
- Regression statistics add MAE, RMSE, mean signed error (prediction minus measurement) and R² (coefficient of determination); residue and matrix tasks also the mean of each protein's MAE, RMSE and signed error.
- `_per_protein_metrics.tsv`: each test sequence's metrics and number of labelled values.
- `_length_metrics.tsv`: test metrics by sequence length (0-49, 50-99, 100-199, 200-399, 400-799, 800+).
- `--no-stats` suppresses both tables as well as `_performance_stats.txt`.
- With a learning-rate schedule, warmup or recovery: `_training_history.tsv` (losses and learning rate per epoch), and `_recovery_events.json` if any recoveries happened.

### Prediction: `parrot-predict` and `Predictor`

- Saved networks record their architecture, hyperparameters, datatype, number of classes and encoding; `parrot-predict` and `Predictor` no longer need these specified.
- Networks saved by earlier versions still load.
- `parrot-predict` writes one output line per input line (keeping every sequence ID) and skips blank and comment lines.
- `parrot-predict --batch` (default 64).
- `parrot-predict` prints its settings only with the new `--verbose`.
- `parrot-predict --silent` prints nothing, even with `--verbose`.
- `parrot-predict` checks `--print-frequency`.
- `parrot-predict` refuses a file that is not a PARROT network, with a message.
- `parrot-predict` checks matrix networks' sequence IDs before predicting.
- `Predictor` accepts `dtype='matrix'`.
- `Predictor.predict_batch()` predicts a list of sequences as `parrot-predict` does.
- `predict_batch()` encodes every sequence first (protein language model embeddings in batches, NARDINI+ features on several cores, into `embedding_cache` or a temporary cache deleted afterwards), then runs the network over batches of similar length.
- `predict_batch()` much faster than `predict()` on each sequence, with the same predictions up to rounding (float32 rounding in the network; for protein language model embeddings, also computed in batches, a stored 16-bit value can move by one step).
- `Predictor` runs on a GPU if there is one (`device='auto'`, as for the commands); PARROT 1's ran on the CPU, which `device='cpu'` still chooses.
- `Predictor(gpu_id=N)` chooses the NVIDIA GPU, as `parrot-predict --gpu-id` does.

### Python API

- The `parrot` package exports only its public names.
- Unused `decode()`, `rev_one_hot()` and `rev_biophysics()` removed.
- `split_data()`, `split_data_cv()` and `vector_split()` check their fractions and number of folds before reading any data.
- `train()` takes the training controls (`gradient_clip_norm`, `check_finite_gradients`, and the rest as `training_options=TrainingOptions(...)`), `control_state=TrainingControlState()` for the learning rates used and any recoveries, and `checkpoint_path`, `resume_from` and `training_log`.

### Other

- Extensive test suite.
- Smaller input-validation and plotting fixes.

## Version 1.7.6 (May 2025)

- Build moved to pyproject.toml and versioningit; compatible with Python 3.13+.

## Version 1.7.2

- Python dependency updated; compatible with Python 3.8, 3.9 and 3.10.
