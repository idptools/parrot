import torch
from parrot.brnn_architecture import BRNN_MtM, BRNN_MtO, ParrotDataModule
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import wandb
import IPython

import pytorch_lightning as pl
# import lightning as L

def objective(trial : optuna.trial.Trial, datamodule : pl.LightningDataModule):
    """Objective function for Optuna to optimize."""

    datatype = datamodule.datatype
    num_classes = datamodule.num_classes
    problem_type = datamodule.problem_type
    input_size = datamodule.input_size

    # Define the hyperparameter search space using trial.suggest_*
    hparams = {
        'hidden_size': trial.suggest_int('hidden_size', 20, 75),        
        'learn_rate': trial.suggest_float('learn_rate', 1e-4, 1e-1,log=True),
        # 'optimizer_name': trial.suggest_categorical('optimizer_name', ['Adam', 'AdamW']),
        'optimizer_name': "SophiaG",
        'beta1': trial.suggest_float('beta1', 0.8, 0.99),
        'beta2': trial.suggest_float('beta2', 0.9, 0.999),
        # 'eps': trial.suggest_float('eps', 1e-8, 1e-1, log=True),
        'rho': trial.suggest_float('eps', 4e-3, 4e-1, log=True),
        'num_layers': trial.suggest_int('num_layers', 1, 2),        
        'input_size': input_size,
        'num_classes': num_classes,
        'problem_type': problem_type,
        'datatype': datatype,
    }
    
    if hparams['optimizer_name'] == 'SGD':
        hparams['momentum'] = trial.suggest_float('momentum', 0.8, 1.0)
        gradient_clip_val = 1.0
    elif hparams['optimizer_name'] == 'AdamW':
        hparams['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
        gradient_clip_val = None
    elif hparams['optimizer_name'] == 'SophiaG':
        hparams['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
        gradient_clip_val = None
    else:
        hparams['momentum'] = None
        gradient_clip_val = None

    print()
    # Print hyperparameters with default values if not defined
    print("learn_rate",hparams.get('learn_rate', 1e-3))
    print("hidden_size",hparams.get('hidden_size'))
    print("num_layers",hparams.get('num_layers'))

    print("beta1",hparams.get('beta1', 0.9))
    print("beta2",hparams.get('beta2', 0.999))
    print("eps",hparams.get('eps', 1e-8))
    
    # print("weight_decay",hparams.get('weight_decay', 0.0))


    # Use a many-to-many architecture


    model = BRNN_MtM(**hparams)
    # compiled_model = torch.compile(model)
    early_stop_callback = EarlyStopping(
                                monitor='average_epoch_val_loss',
                                min_delta=0.000,
                                patience=5,
                                verbose=False,
                                mode='min'
                                )

    wandb_logger = WandbLogger(name=f"run{trial.number}",
                               project='metapredict_b64_SophiaG')
    
    wandb_logger.watch(model)

    # gradient_clip_val = 1.0
    trainer = pl.Trainer(
        gradient_clip_val=gradient_clip_val,
        precision="16-mixed",  
        logger=wandb_logger,
        enable_checkpointing=True,
        # limit_train_batches=0.3,
        max_epochs=50,
        accelerator="auto",
        devices=[1],
        callbacks = [PyTorchLightningPruningCallback(trial, monitor="average_epoch_val_loss"),early_stop_callback],
    )
    trainer.logger.log_hyperparams(hparams)

    trainer.fit(model, datamodule=datamodule)
    
    wandb_logger.experiment.unwatch(model)
    wandb.finish()
    # Return the validation loss as the objective value for Optuna
    return trainer.callback_metrics['average_epoch_val_loss'].detach()


datamodule = ParrotDataModule("meta.tsv",
                              num_classes=1,
                              datatype="residues", 
                              split_file="meta_2023_06_28_split_file.txt",
                              ignore_warnings=True,
                              batch_size=64)

# this can improve performance for tensor cores cards - should figure out how to do this more elegantly.
torch.set_float32_matmul_precision("high")

study_name = "metapredict_b64_SophiaG"
storage = f"sqlite:///{study_name}.db"

pruner = optuna.pruners.MedianPruner()

# quasi-random search is recommended by google because its more efficient at search than random
# but isnt pigeon holed by adaptive sampling approach like TPE (e.g., if you want to rerank based
# on another metric you can)

# sampler = optuna.samplers.QMCSampler(qmc_type="halton", scramble=True,seed=42)
sampler = optuna.samplers.TPESampler()


study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage, 
                            direction='minimize', pruner=pruner,load_if_exists=True)

study.optimize(lambda trial: objective(trial, datamodule), n_trials=100)

# Print the best hyperparameters and objective value
best_trial = study.best_trial
print(f"Best trial - Loss: {best_trial.value:.4f}")
print("Best trial - Hyperparameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")



