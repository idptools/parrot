"""
Core training module of PARROT

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold

from parrot import brnn_plot
from parrot import encode_sequence
from parrot.unet_architecture import UNet_PARROT
from parrot.process_input_data import SequenceDataset, seq_regress_collate, seq_class_collate, res_regress_collate, res_class_collate
from parrot.encode_sequence import ParrotLightningEncoder


def matrix_collate(batch):
    """Collate function for matrix data (UNet)"""
    # Extract names, matrices, and targets from the batch
    names = [item[0] for item in batch]
    matrices = [item[1].clone().detach().float() for item in batch]
    targets = [item[2] for item in batch]
    
    # Stack matrices into a batch tensor
    # Assumes matrices are already in the correct shape (height, width, channels)
    # For UNet, we need (batch, channels, height, width)
    batch_matrices = torch.stack(matrices, dim=0)
    
    # Handle targets based on their structure
    if isinstance(targets[0], (int, float)):
        # Single value per matrix (classification/regression)
        targets_tensor = torch.tensor(targets, dtype=torch.float32 if isinstance(targets[0], float) else torch.long)
    elif isinstance(targets[0], torch.Tensor):
        # Targets are already tensors, just handle dimensionality
        if targets[0].dim() == 0:  # Scalar tensors
            targets_tensor = torch.stack(targets, dim=0)
        else:
            # Matrix of values (residue-level for matrices)
            targets_tensor = torch.stack(targets, dim=0)
    else:
        # Matrix of values (residue-level for matrices)
        targets_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets], dim=0)
    
    return names, batch_matrices, targets_tensor


def get_collate_function(datatype, problem_type):
    """Get the appropriate collate function based on datatype and problem_type"""
    if datatype == 'matrix':
        return matrix_collate
    elif datatype == 'sequence':
        if problem_type == 'regression':
            return seq_regress_collate
        else:
            return seq_class_collate
    elif datatype == 'residues':
        if problem_type == 'regression':
            return res_regress_collate
        else:
            return res_class_collate
    else:
        raise ValueError(f"Unknown datatype: {datatype}")


def train(network, train_dataset, val_dataset, datatype, problem_type, weights_file,
          stop_condition, device, learn_rate, n_epochs, verbose=False, silent=False,
          batch_size=32, encoder_cfg=None, cross_validation=False, cv_folds=5):
    """Train a PARROT network (BRNN or UNet) and save the best performing network weights

    Train the network on a training set, and every epoch evaluate its performance on
    a validation set. Save the network weights that achieve the best performance on
    the validation set.

    This function now supports both traditional BRNN training and new Lightning-based
    UNet training. It automatically detects the network type and uses appropriate
    training procedures.

    Parameters
    ----------
    network : PyTorch network object or Lightning module
            A BRNN network or UNet_PARROT with the desired architecture
    train_dataset : SequenceDataset or DataLoader
            Training dataset containing sequences and targets
    val_dataset : SequenceDataset or DataLoader  
            Validation dataset containing sequences and targets
    datatype : str
            The format of values in the dataset. Should be 'sequence' for datasets
            with a single value per sequence, 'residues' for datasets with values
            for every residue, or 'matrix' for UNet matrix inputs.
    problem_type : str
            The machine learning task--should be either 'regression' or
            'classification'.
    weights_file : str
            A path to the location where the best_performing network weights will be
            saved
    stop_condition : str
            Determines when to conclude network training. If 'iter', then the network
            will train for `n_epochs` epochs, then stop. If 'auto' then the network
            will train for at least `n_epochs` epochs, then begin assessing whether
            performance has sufficiently stagnated.
    device : str
            Location of where training will take place--should be either 'cpu' or
            'cuda' (GPU). If available, training on GPU is typically much faster.
    learn_rate : float
            Initial learning rate of network training.
    n_epochs : int
            Number of epochs to train for.
    verbose : bool, optional
            If true, causes training updates to be written every epoch.
    silent : bool, optional
            If true, causes no training updates to be written to standard out.
    batch_size : int, optional
            Batch size for training. Default is 32.
    encoder_cfg : DictConfig, optional
            Configuration for the encoder (for UNet training).
    cross_validation : bool, optional
            Whether to perform cross-validation. Only supported with Lightning modules.
    cv_folds : int, optional
            Number of folds for cross-validation. Default is 5.

    Returns
    -------
    list or dict
            For regular training: A list of the average training set losses and validation losses
            For cross-validation: A dictionary with CV results and fold performances
    """
    # Check if this is a Lightning module (UNet) or traditional network (BRNN)
    is_lightning_module = isinstance(network, L.LightningModule)
    
    if is_lightning_module and datatype == 'matrix':
        # Use Lightning training for UNet
        return _train_lightning_unet(network, train_dataset, val_dataset, datatype, 
                                   problem_type, weights_file, stop_condition, device,
                                   learn_rate, n_epochs, verbose, silent, batch_size,
                                   cross_validation, cv_folds)
    else:
        # Use traditional training for BRNN
        return _train_traditional_brnn(network, train_dataset, val_dataset, datatype,
                                     problem_type, weights_file, stop_condition, device,
                                     learn_rate, n_epochs, verbose, silent, batch_size,
                                     cross_validation, cv_folds)


def _train_lightning_unet(network, train_dataset, val_dataset, datatype, problem_type,
                         weights_file, stop_condition, device, learn_rate, n_epochs,
                         verbose, silent, batch_size, cross_validation, cv_folds):
    """Train UNet using PyTorch Lightning"""

    # Check if cross-validation is enabled
    if cross_validation:
        return _train_with_cross_validation(network, train_dataset, val_dataset, datatype,
                                          problem_type, weights_file, device, learn_rate,
                                          n_epochs, verbose, silent, batch_size, cv_folds)
    
    # Setup data loaders
    collate_fn = get_collate_function(datatype, problem_type)
    
    if isinstance(train_dataset, DataLoader):
        train_loader = train_dataset
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_fn)
    
    if isinstance(val_dataset, DataLoader):
        val_loader = val_dataset
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.dirname(weights_file),
        filename=os.path.basename(weights_file).replace('.pt', ''),
        monitor='epoch_val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping if auto stop condition
    #WARNING: This does not just perform training for n_epochs, but rather
    # stops training if performance has not improved for n_epochs epochs.
    # This condition only applies if you do not reach the max_epochs limit.
    if stop_condition == 'auto':
        early_stop_callback = EarlyStopping(
            monitor='epoch_val_loss',
            patience=n_epochs,
            mode='min',
            verbose=not silent
        )
        callbacks.append(early_stop_callback)
    
    # Setup trainer
    accelerator = 'gpu' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
    max_epochs = n_epochs if stop_condition == 'iter' else 1000  # Large number for auto stopping
    
    logger = None if silent else TensorBoardLogger("logs", name="parrot_training")
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        log_every_n_steps=1 if verbose else 50
    )
    
    # Train the model
    trainer.fit(network, train_loader, val_loader)
    
    # Load best weights and save in the requested format
    best_model = network.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.state_dict(), weights_file)
    
    # Extract training history
    train_losses = []
    val_losses = []
    
    # Get logged metrics from trainer if available
    if hasattr(trainer, 'logged_metrics'):
        for epoch_metrics in trainer.logged_metrics:
            if 'epoch_train_loss' in epoch_metrics:
                train_losses.append(epoch_metrics['epoch_train_loss'])
            if 'epoch_val_loss' in epoch_metrics:
                val_losses.append(epoch_metrics['epoch_val_loss'])
    
    return train_losses, val_losses


def _train_with_cross_validation(network, full_dataset, val_dataset, datatype, problem_type,
                                weights_file, device, learn_rate, n_epochs, verbose, silent,
                                batch_size, cv_folds):
    """Perform cross-validation training using Lightning"""
    
    # Combine train and val datasets for CV splitting
    if isinstance(full_dataset, SequenceDataset) and isinstance(val_dataset, SequenceDataset):
        # Combine the data from both datasets
        combined_data = full_dataset.data + val_dataset.data
        # Create new combined dataset
        combined_dataset = SequenceDataset.__new__(SequenceDataset)
        combined_dataset.data = combined_data
        combined_dataset.encoder = full_dataset.encoder
        combined_dataset.datatype = full_dataset.datatype
    else:
        # Use the training dataset for CV
        combined_dataset = full_dataset
    
    # Setup KFold
    # WARNING: random state is set to 42 for reproducibility, but this may not be suitable for all use cases
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42) 
    
    cv_results = {
        'fold_train_losses': [],
        'fold_val_losses': [],
        'fold_final_val_loss': [],
        'mean_val_loss': 0.0,
        'std_val_loss': 0.0
    }
    # Get indices for the dataset
    dataset_indices = list(range(len(combined_dataset)))

    # Loop through each fold
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset_indices)):
        if not silent:
            print(f"Training fold {fold + 1}/{cv_folds}")
        
        # Create fold datasets
        fold_train_data = [combined_dataset.data[i] for i in train_indices]
        fold_val_data = [combined_dataset.data[i] for i in val_indices]
        
        # Create new dataset objects for this fold
        fold_train_dataset = SequenceDataset.__new__(SequenceDataset)
        fold_train_dataset.data = fold_train_data
        fold_train_dataset.encoder = combined_dataset.encoder
        fold_train_dataset.datatype = combined_dataset.datatype
        
        fold_val_dataset = SequenceDataset.__new__(SequenceDataset)
        fold_val_dataset.data = fold_val_data
        fold_val_dataset.encoder = combined_dataset.encoder
        fold_val_dataset.datatype = combined_dataset.datatype
        
        # Create a new model instance for this fold
        fold_network = type(network)(
            input_channels=network.input_channels,
            num_classes=network.num_classes,
            problem_type=network.problem_type,
            batch_size=network.batch_size,
            **{k: v for k, v in network.hparams.items() 
               if k not in ['input_channels', 'num_classes', 'problem_type', 'batch_size']}
        )
        
        # Train this fold
        fold_weights_file = weights_file.replace('.pt', f'_fold_{fold + 1}.pt')
        train_losses, val_losses = _train_lightning_unet(
            fold_network, fold_train_dataset, fold_val_dataset, datatype, problem_type,
            fold_weights_file, 'iter', device, learn_rate, n_epochs, False, silent,
            batch_size, False, cv_folds
        )
        
        # Store results
        cv_results['fold_train_losses'].append(train_losses)
        cv_results['fold_val_losses'].append(val_losses)
        cv_results['fold_final_val_loss'].append(val_losses[-1] if val_losses else float('inf'))
    
    # Calculate statistics
    final_val_losses = cv_results['fold_final_val_loss']
    cv_results['mean_val_loss'] = np.mean(final_val_losses)
    cv_results['std_val_loss'] = np.std(final_val_losses)
    
    if not silent:
        print(f"Cross-validation complete: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
    
    # Save the model from the best fold
    best_fold = np.argmin(final_val_losses)
    best_fold_weights = weights_file.replace('.pt', f'_fold_{best_fold + 1}.pt')
    if os.path.exists(best_fold_weights):
        import shutil
        shutil.copy2(best_fold_weights, weights_file)
    
    return cv_results


def _train_traditional_brnn(network, train_dataset, val_dataset, datatype, problem_type,
                           weights_file, stop_condition, device, learn_rate, n_epochs,
                           verbose, silent, batch_size, cross_validation=False, cv_folds=5):
    """Traditional BRNN training (original implementation)"""
    
    # Check if cross-validation is enabled
    if cross_validation:
        return _train_brnn_with_cross_validation(network, train_dataset, val_dataset, datatype,
                                               problem_type, weights_file, stop_condition, device,
                                               learn_rate, n_epochs, verbose, silent, batch_size, cv_folds)
    
    # Setup data loaders if not already provided
    collate_fn = get_collate_function(datatype, problem_type)
    
    if isinstance(train_dataset, DataLoader):
        train_loader = train_dataset
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn)
    
    if isinstance(val_dataset, DataLoader):
        val_loader = val_dataset
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)

    # Set optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learn_rate)

    # Set loss criteria
    if problem_type == 'regression':
        if datatype == 'residues':
            criterion = nn.MSELoss(reduction='sum')
        elif datatype == 'sequence':
            criterion = nn.L1Loss(reduction='sum')
    elif problem_type == 'classification':
        criterion = nn.CrossEntropyLoss(reduction='sum')

    network = network.float()
    total_step = len(train_loader)
    min_val_loss = np.inf
    avg_train_losses = []
    avg_val_losses = []

    if stop_condition == 'auto':
        min_epochs = n_epochs
        # Set to some arbitrarily large number of iterations -- will stop automatically
        n_epochs = 20000000
        last_decrease = 0

    # Train the model - evaluate performance on val set every epoch
    end_training = False
    for epoch in range(n_epochs):  # Main loop

        # Initialize training and testing loss for epoch
        train_loss = 0
        val_loss = 0

        # Iterate over batches
        for i, (names, vectors, targets) in enumerate(train_loader):
            vectors = vectors.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = network(vectors.float())

            if problem_type == 'regression':
                loss = criterion(outputs, targets.float())
            else:
                if datatype == 'residues':
                    outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, targets.long())

            train_loss += loss.data.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for names, vectors, targets in val_loader:
            vectors = vectors.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = network(vectors.float())
            if problem_type == 'regression':
                loss = criterion(outputs, targets.float())
            else:
                if datatype == 'residues':
                    outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, targets.long())

            # Increment val loss
            val_loss += loss.data.item()

        # Avg loss:
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        signif_decrease = True
        if stop_condition == 'auto' and epoch > min_epochs - 1:
            # Check to see if loss has stopped decreasing
            last_epochs_loss = avg_val_losses[-min_epochs:]

            for loss in last_epochs_loss:
                if val_loss >= loss*0.995:
                    signif_decrease = False

            # If network performance has plateaued over the last range of epochs, end training
            if not signif_decrease and epoch - last_decrease > min_epochs:
                end_training = True

        # Only save updated weights to memory if they improve val set performance
        if val_loss < min_val_loss:
            min_val_loss = val_loss 	# Reset min_val_loss
            last_decrease = epoch
            torch.save(network.state_dict(), weights_file)  # Save model

        # Append losses to lists
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)

        if verbose:
            print('Epoch %d\tLoss %.4f' % (epoch, val_loss))
        elif epoch % 5 == 0 and silent is False:
            print('Epoch %d\tLoss %.4f' % (epoch, val_loss))

        # This is placed here to ensure that the best network, even if the performance
        # improvement is marginal, is saved.
        if end_training:
            break

    # Return loss per epoch so that they can be plotted
    return avg_train_losses, avg_val_losses


def _train_brnn_with_cross_validation(network, train_dataset, val_dataset, datatype, problem_type,
                                     weights_file, stop_condition, device, learn_rate, n_epochs,
                                     verbose, silent, batch_size, cv_folds):
    """Perform cross-validation training for traditional BRNN"""
    
    # Import copy to create network instances for each fold
    import copy
    
    # Combine train and val datasets for CV splitting
    if isinstance(train_dataset, SequenceDataset) and isinstance(val_dataset, SequenceDataset):
        # Combine the data from both datasets
        combined_data = train_dataset.data + val_dataset.data
        # Create new combined dataset
        combined_dataset = SequenceDataset.__new__(SequenceDataset)
        combined_dataset.data = combined_data
        combined_dataset.encoder = train_dataset.encoder
        combined_dataset.datatype = train_dataset.datatype
    else:
        # Use the training dataset for CV
        combined_dataset = train_dataset
    
    # Setup KFold
    # WARNING: random state is set to 42 for reproducibility, but this may not be suitable for all use cases
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'fold_train_losses': [],
        'fold_val_losses': [],
        'fold_final_val_loss': [],
        'mean_val_loss': 0.0,
        'std_val_loss': 0.0
    }
    
    # Get indices for the dataset
    dataset_indices = list(range(len(combined_dataset)))
    
    # Loop through each fold
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset_indices)):
        if not silent:
            print(f"Training fold {fold + 1}/{cv_folds}")
        
        # Create fold datasets
        fold_train_data = [combined_dataset.data[i] for i in train_indices]
        fold_val_data = [combined_dataset.data[i] for i in val_indices]
        
        # Create new dataset objects for this fold
        fold_train_dataset = SequenceDataset.__new__(SequenceDataset)
        fold_train_dataset.data = fold_train_data
        fold_train_dataset.encoder = combined_dataset.encoder
        fold_train_dataset.datatype = combined_dataset.datatype
        
        fold_val_dataset = SequenceDataset.__new__(SequenceDataset)
        fold_val_dataset.data = fold_val_data
        fold_val_dataset.encoder = combined_dataset.encoder
        fold_val_dataset.datatype = combined_dataset.datatype
        
        # Create a new network instance for this fold (deep copy to avoid weight sharing)
        fold_network = copy.deepcopy(network)
        
        # Train this fold
        fold_weights_file = weights_file.replace('.pt', f'_fold_{fold + 1}.pt')
        train_losses, val_losses = _train_traditional_brnn(
            fold_network, fold_train_dataset, fold_val_dataset, datatype, problem_type,
            fold_weights_file, stop_condition, device, learn_rate, n_epochs, False, silent,
            batch_size, False, cv_folds  # cross_validation=False to avoid recursion
        )
        
        # Store results
        cv_results['fold_train_losses'].append(train_losses)
        cv_results['fold_val_losses'].append(val_losses)
        cv_results['fold_final_val_loss'].append(val_losses[-1] if val_losses else float('inf'))
    
    # Calculate statistics
    final_val_losses = cv_results['fold_final_val_loss']
    cv_results['mean_val_loss'] = np.mean(final_val_losses)
    cv_results['std_val_loss'] = np.std(final_val_losses)
    
    if not silent:
        print(f"Cross-validation complete: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
    
    # Save the model from the best fold
    best_fold = np.argmin(final_val_losses)
    best_fold_weights = weights_file.replace('.pt', f'_fold_{best_fold + 1}.pt')
    if os.path.exists(best_fold_weights):
        import shutil
        shutil.copy2(best_fold_weights, weights_file)
    
    return cv_results


def test_labeled_data(network, test_dataset, datatype,
                      problem_type, weights_file, num_classes,
                      probabilistic_classification, include_figs, 
                      device, output_file_prefix='', batch_size=32):
    """Test a trained PARROT network (BRNN or UNet) on labeled sequences

    Using the saved weights of a trained network, run a set of sequences through
    the network and evaluate the performance. Return the average loss per
    sequence and plot the results. Testing a network on previously-unseen data 
    provides a useful estimate of how generalizable the network's performance is.

    This function now supports both traditional BRNN testing and new Lightning-based
    UNet testing. It automatically detects the network type and uses appropriate
    testing procedures.

    Parameters
    ----------
    network : PyTorch network object or Lightning module
            A BRNN network or UNet_PARROT with the desired architecture
    test_dataset : SequenceDataset or DataLoader
            Dataset containing the sequences and targets of the test set
    datatype : str
            The format of values in the dataset. Should be 'sequence' for datasets
            with a single value per sequence, 'residues' for datasets with values
            for every residue, or 'matrix' for UNet matrix inputs.
    problem_type : str
            The machine learning task--should be either 'regression' or
            'classification'.
    weights_file : str
            A path to the location of the best_performing network weights
    num_classes: int
            Number of data classes. If regression task, put 1.
    probabilistic_classification: bool
            Whether output should be binary labels, or "weights" of each label type.
            This field is only implemented for binary, sequence classification tasks.
    include_figs: bool
            Whether or not matplotlib figures should be generated.
    device : str
            Location of where testing will take place--should be either 'cpu' or
            'cuda' (GPU). If available, training on GPU is typically much faster.
    output_file_prefix : str
            Path and filename prefix to which the test set predictions and plots will be saved.
    batch_size : int, optional
            Batch size for testing. Default is 32.

    Returns
    -------
    float
            The average loss across the entire test set
    list of lists
            Details of the output predictions for each of the sequences in the test set. Each
            inner list represents a sample in the test set, with the format: [sequence_vector,
            true_value, predicted_value, sequence_ID]
    """
    # Check if this is a Lightning module (UNet) or traditional network (BRNN)
    is_lightning_module = isinstance(network, L.LightningModule)
    
    if is_lightning_module and datatype == 'matrix':
        # Use Lightning testing for UNet
        return _test_lightning_unet(network, test_dataset, datatype, problem_type,
                                  weights_file, num_classes, probabilistic_classification,
                                  include_figs, device, output_file_prefix, batch_size)
    else:
        # Use traditional testing for BRNN
        return _test_traditional_brnn(network, test_dataset, datatype, problem_type,
                                    weights_file, num_classes, probabilistic_classification,
                                    include_figs, device, output_file_prefix, batch_size)


def _test_lightning_unet(network, test_dataset, datatype, problem_type, weights_file,
                        num_classes, probabilistic_classification, include_figs, device,
                        output_file_prefix, batch_size):
    """Test UNet using PyTorch Lightning"""
    
    # Load network weights
    network.load_state_dict(torch.load(weights_file))
    network.eval()
    
    # Setup data loader
    collate_fn = get_collate_function(datatype, problem_type)
    
    if isinstance(test_dataset, DataLoader):
        test_loader = test_dataset
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn)
    
    # Setup trainer for testing
    accelerator = 'gpu' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
    trainer = L.Trainer(accelerator=accelerator, logger=False, enable_progress_bar=False)
    
    # Run test
    test_results = trainer.test(network, test_loader, verbose=False)
    test_loss = test_results[0].get('test_loss', 0.0)
    
    # Collect predictions for detailed analysis
    predictions = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for names, matrices, targets in test_loader:
            matrices = matrices.to(device)
            targets = targets.to(device)
            
            outputs = network(matrices.float())
            
            all_targets.append(targets)
            all_outputs.append(outputs.detach())
            
            # Store predictions for each sample
            for i in range(len(names)):
                predictions.append([
                    matrices[i].cpu().numpy(),
                    targets[i].cpu().numpy(),
                    outputs[i].cpu().detach().numpy(),
                    names[i]
                ])
    
    # Generate plots if requested
    if include_figs:
        _generate_unet_plots(all_targets, all_outputs, problem_type, datatype,
                           num_classes, output_file_prefix)
    
    return test_loss, predictions


def _test_traditional_brnn(network, test_dataset, datatype, problem_type, weights_file,
                          num_classes, probabilistic_classification, include_figs, device,
                          output_file_prefix, batch_size):
    """Traditional BRNN testing (original implementation)"""
    
    # Load network weights
    network.load_state_dict(torch.load(weights_file))

    # Get output directory for images
    network_filename = weights_file.split('/')[-1]
    output_dir = weights_file[:-len(network_filename)]

    # Setup data loader
    collate_fn = get_collate_function(datatype, problem_type)
    
    if isinstance(test_dataset, DataLoader):
        test_loader = test_dataset
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,  # batch size 1 for compatibility
                               collate_fn=collate_fn)

    # Set loss criteria
    if problem_type == 'regression':
        criterion = nn.MSELoss()
    elif problem_type == 'classification':
        criterion = nn.CrossEntropyLoss()

    test_loss = 0
    all_targets = []
    all_outputs = []
    predictions = []
    for names, vectors, targets in test_loader: 	# batch size of 1
        all_targets.append(targets)

        vectors = vectors.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = network(vectors.float())
        if problem_type == 'regression':
            loss = criterion(outputs, targets.float())
        else:
            if datatype == 'residues':
                outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, targets.long())

        test_loss += loss.data.item()  # Increment test loss
        all_outputs.append(outputs.detach())

        # Add to list as: [seq_vector, true value, predicted value, name]
        if len(targets.shape) > 0 and targets.shape[0] == 1:
            # Handle single-item batches
            target_value = targets.cpu().numpy()[0] if targets.numel() > 1 else targets.cpu().numpy().item()
        else:
            target_value = targets.cpu().numpy()
            
        predictions.append([vectors[0].cpu().numpy(), target_value,
                           outputs.cpu().detach().numpy(), names[0]])

    # Plot 'accuracy' depending on the problem type and datatype
    if problem_type == 'regression':
        if datatype == 'residues':
            if include_figs:
                brnn_plot.residue_regression_scatterplot(all_targets, all_outputs, 
                                            output_file_prefix=output_file_prefix)

            # Format predictions
            for i in range(len(predictions)):
                predictions[i][2] = predictions[i][2].flatten()
                predictions[i][1] = predictions[i][1].flatten()

        elif datatype == 'sequence':
            if include_figs:
                brnn_plot.sequence_regression_scatterplot(all_targets, all_outputs, 
                                            output_file_prefix=output_file_prefix)

            # Format predictions
            for i in range(len(predictions)):
                predictions[i][2] = predictions[i][2][0][0] if predictions[i][2].size > 1 else predictions[i][2].item()
                if isinstance(predictions[i][1], np.ndarray) and predictions[i][1].size == 1:
                    predictions[i][1] = predictions[i][1].item()
                elif isinstance(predictions[i][1], np.ndarray):
                    predictions[i][1] = predictions[i][1][0] if len(predictions[i][1]) > 0 else predictions[i][1]

    elif problem_type == 'classification':

        if datatype == 'residues':
            if include_figs:
                brnn_plot.res_confusion_matrix(all_targets, all_outputs, num_classes, 
                                            output_file_prefix=output_file_prefix)

            # Format predictions and assign class predictions
            for i in range(len(predictions)):
                pred_values = []
                for j in range(len(predictions[i][2])):
                    pred_values = np.argmax(predictions[i][2], axis=1)[0]
                predictions[i][2] = np.array(pred_values, dtype=np.int)

        elif datatype == 'sequence':
            if probabilistic_classification:
                # Probabilistic assignment of class predictions
                # Optional implementation for classification tasks
                # e.g. every sequence is assigned probabilities
                # corresponding to each possible class
                pred_probabilities = []
                for i in range(len(predictions)):
                    softmax = np.exp(predictions[i][2][0])
                    probs = softmax / np.sum(softmax)
                    predictions[i][2] = probs
                    pred_probabilities.append(probs)

                # Plot ROC and PR curves
                if include_figs:
                    brnn_plot.plot_roc_curve(all_targets, pred_probabilities, num_classes, 
                                            output_file_prefix=output_file_prefix)
                    brnn_plot.plot_precision_recall_curve(all_targets, pred_probabilities, 
                                            num_classes, output_file_prefix=output_file_prefix)

            else:
                # Absolute assignment of class predictions
                # e.g. every sequence receives an integer class label
                for i in range(len(predictions)):
                    pred_value = np.argmax(predictions[i][2])
                    predictions[i][2] = int(pred_value)

                # Plot confusion matrix (if not in probabilistic classification mode)
                if include_figs:
                    brnn_plot.confusion_matrix(all_targets, all_outputs, num_classes, 
                                                output_file_prefix=output_file_prefix)

    return test_loss / len(test_loader.dataset), predictions


def _generate_unet_plots(all_targets, all_outputs, problem_type, datatype, num_classes, output_file_prefix):
    """Generate plots for UNet results"""
    # For now, use existing plotting functions
    # In the future, specialized UNet plotting functions could be added
    if problem_type == 'regression':
        brnn_plot.sequence_regression_scatterplot(all_targets, all_outputs, 
                                    output_file_prefix=output_file_prefix)
    elif problem_type == 'classification':
        brnn_plot.confusion_matrix(all_targets, all_outputs, num_classes, 
                                    output_file_prefix=output_file_prefix)


def test_unlabeled_data(network, sequences, device, encoding_scheme='onehot', encoder=None, print_frequency=None):
    """Test a trained PARROT network (BRNN or UNet) on unlabeled sequences

    Use a trained network to make predictions on previously-unseen data.

    ** 
    Note: Unlike the previous functions, `network` here must have pre-loaded
    weights for BRNN networks. For UNet networks, weights should be loaded separately.
    **

    Parameters
    ----------
    network : PyTorch network object or Lightning module
            A BRNN network or UNet_PARROT with the desired architecture and pre-loaded weights
    sequences : list
            A list of amino acid sequences to test using the network
    device : str
            Location of where testing will take place--should be either 'cpu' or
            'cuda' (GPU). If available, training on GPU is typically much faster.
    encoding_scheme : str, optional
            How amino acid sequences are to be encoded as numeric vectors. Currently,
            'onehot','biophysics' and 'user' are the implemented options.
    encoder: UserEncoder object, optional
            If encoding_scheme is 'user', encoder should be a UserEncoder object
            that can convert amino acid sequences to numeric vectors. If
            encoding_scheme is not 'user', use None.
    print_frequency : int
            If provided defines at what sequence interval an update is printed.
            Default = None.
    
    Returns
    -------
    dict
            A dictionary containing predictions mapped to sequences
    """
    # Check if this is a Lightning module (UNet)
    is_lightning_module = isinstance(network, L.LightningModule)
    
    if is_lightning_module:
        # For UNet, we need matrix input, which is not directly supported
        # by the sequences input format. This would need specialized handling.
        raise NotImplementedError("test_unlabeled_data for UNet requires matrix input, not sequences")
    
    # Traditional BRNN prediction
    pred_dict = {}

    local_count = -1
    total_count = len(sequences)

    for seq in sequences:

        local_count = local_count + 1
        if print_frequency is not None:
            if local_count % print_frequency == 0:
                print(f'On {local_count} of {total_count}')

        if encoding_scheme == 'onehot':
            seq_vector = encode_sequence.one_hot(seq)
        elif encoding_scheme == 'biophysics':
            seq_vector = encode_sequence.biophysics(seq)
        elif encoding_scheme == 'user':
            seq_vector = encoder.encode(seq)

        seq_vector = seq_vector.view(1, len(seq_vector), -1)

        # Forward pass
        outputs = network(seq_vector.float()).detach().numpy()
        pred_dict[seq] = outputs

    return pred_dict
