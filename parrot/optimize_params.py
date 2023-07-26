import argparse
import yaml
import os
from parrot import get_directory

import torch
from parrot.brnn_architecture import BRNN_MtM, BRNN_MtO, ParrotDataModule
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

from pytorch_lightning.loggers import WandbLogger
import wandb
import IPython

import pytorch_lightning as pl
# import lightning as L

def determine_matmul_precision():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return torch.cuda.get_device_properties(device).major >= 7
    else: 
        return False
    

def objective(trial : optuna.trial.Trial, datamodule : pl.LightningDataModule, config):
    """Objective function for Optuna to optimize."""

    datatype = datamodule.datatype
    num_classes = datamodule.num_classes
    problem_type = datamodule.problem_type
    input_size = datamodule.input_size

    # Define the hyperparameter search space using trial.suggest_*
    hparams = {
<<<<<<< HEAD
        config.optimizer_name['name']: trial.suggest_categorical(config.optimizer_name['name'],
                                                               list(config.optimizer_name['choices'].values())
                                                               ), 
        
        config.num_lstm_layers['name'] : trial.suggest_int(config.num_lstm_layers['name'],
                                                         config.num_lstm_layers['min'],
                                                         config.num_lstm_layers['max']),

        config.lstm_hidden_size['name'] : trial.suggest_int(config.lstm_hidden_size['name'],
                                                           config.lstm_hidden_size['min'],
                                                            config.lstm_hidden_size['max']),

        config.learn_rate['name'] : trial.suggest_float(config.learn_rate['name'],
                                                      config.learn_rate['min'],
                                                      config.learn_rate['max'],
                                                      log=config.learn_rate['log']
                                                      ),

=======
        'hidden_size': trial.suggest_int('hidden_size', 45, 50),        
        #'learn_rate': trial.suggest_float('learn_rate', 1e-4, 1e-1,log=True),
        #'hidden_size': 75,
        'learn_rate': trial.suggest_float('learn_rate', 1e-3, 1e-2,log=True),
        #'learn_rate': 0.00495508214936704,
        # 'optimizer_name': trial.suggest_categorical('optimizer_name', ['Adam', 'AdamW']),
        'optimizer_name': "SGD",
        #'beta1': trial.suggest_float('beta1', 0.8, 0.99),
        #'beta2': trial.suggest_float('beta2', 0.9, 0.999),
        #'eps': trial.suggest_float('eps', 1e-8, 1e-1, log=True),
        #'rho': trial.suggest_float('eps', 4e-3, 4e-1, log=True),
        #'num_layers': trial.suggest_int('num_layers', 1, 2),        
        'num_layers': 2,
>>>>>>> 1c823d06dfc6fbcfe96f753a39f2a173f4bf828b
        'input_size': input_size,
        'num_classes': num_classes,
        'problem_type': problem_type,
        'datatype': datatype,
    }
    
    num_linear_layers = trial.suggest_int(config.num_linear_layers['name'],
                      config.num_linear_layers['min'],
                      config.num_linear_layers['max'])
    
    if num_linear_layers > 1:
        hparams[config.num_linear_layers['name']] = num_linear_layers
        
        hparams[config.linear_hidden_size['name']] = trial.suggest_int(config.linear_hidden_size['name'],
                                                            config.linear_hidden_size['min'],
                                                            config.linear_hidden_size['max'])
        
        hparams[config.dropout['name']] : trial.suggest_float(config.dropout['name'],
                                                   config.dropout['min'],
                                                   config.dropout['max'])


    if hparams['optimizer_name'] == 'SGD':
        hparams['momentum'] = trial.suggest_float('momentum', 0.98, 1.0)
        #hparams['momentum'] = 0.9972361251129012
        gradient_clip_val = 1.0
    elif hparams['optimizer_name'] == 'AdamW':
        hparams['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
        gradient_clip_val = None
    else:
        hparams['momentum'] = None
        gradient_clip_val = None

    print()
    # Print hyperparameters with default values if not defined
    print("learn_rate",hparams.get('learn_rate', 1e-3))
<<<<<<< HEAD
    print("lstm_hidden_size",hparams.get('lstm_hidden_size'))
    print("num_lstm_layers",hparams.get('num_lstm_layers'))
    print("linear_hidden_size",hparams.get('linear_hidden_size'),0)
    print("num_linear_layers",hparams.get('num_linear_layers'),1)
    print("dropout",hparams.get('dropout'))
=======
    print("hidden_size",hparams.get('hidden_size'))
    print("num_layers",hparams.get('num_layers'))

    print("momentum",hparams.get('momentum', 0.9))
>>>>>>> 1c823d06dfc6fbcfe96f753a39f2a173f4bf828b
    print("beta1",hparams.get('beta1', 0.9))
    print("beta2",hparams.get('beta2', 0.999))
    print("eps",hparams.get('eps', 1e-8))

    if datatype == 'sequence':
        model = BRNN_MtO(**hparams)
    elif datatype == 'residues':
        model = BRNN_MtM(**hparams)

    early_stop_callback = EarlyStopping(
                                monitor='average_epoch_val_loss',
                                min_delta=0.00,
                                patience=3,
                                verbose=False,
                                mode='min'
                                )

<<<<<<< HEAD
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="average_epoch_val_loss")
    
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    wandb_logger = WandbLogger(name=f"run{trial.number}",
                               project='metapredict_b64_mlp')
=======
    # could play with the swa_lr as a tunable parameter too but :shrug: just fiddling with stuff for now
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    wandb_logger = WandbLogger(name=f"run{trial.number}",
                               project='metapredict_layer_norm_b256')
>>>>>>> 1c823d06dfc6fbcfe96f753a39f2a173f4bf828b
    
    wandb_logger.watch(model)

    # gradient_clip_val = 1.0
    trainer = pl.Trainer(
        gradient_clip_val=gradient_clip_val,
        precision="16-mixed",  
        logger=wandb_logger,
        enable_checkpointing=True,
<<<<<<< HEAD
        # limit_train_batches=5,
        max_epochs=50,
        accelerator="auto",
        devices=[int(config.gpu_id)],
        callbacks = [pruning_callback,
                     early_stop_callback],
=======
        # limit_train_batches=0.3,
        max_epochs=100,
        accelerator="auto",
        #devices="auto",
        #devices=[0],
        #devices=[1],
        #devices=[2],
        devices=[3],
        callbacks = [PyTorchLightningPruningCallback(trial, monitor="average_epoch_val_loss"),early_stop_callback,swa_callback],
        #callbacks = [PyTorchLightningPruningCallback(trial, monitor="average_epoch_val_loss"),early_stop_callback],
>>>>>>> 1c823d06dfc6fbcfe96f753a39f2a173f4bf828b
    )
    trainer.logger.log_hyperparams(hparams)

    trainer.fit(model, datamodule=datamodule)
    
    wandb_logger.experiment.unwatch(model)
    wandb.finish()
    # Return the validation loss as the objective value for Optuna
    return trainer.callback_metrics['average_epoch_val_loss'].detach()

def run_optimization(config,    
                     study_name,
                     tsv_file,
                     split_file, 
                     num_classes,
                     datatype, 
                     batch_size,
                     n_trials=100,
                     ignore_warnings=False):
    
    # this can improve performance for tensor cores cards
    if determine_matmul_precision():
        torch.set_float32_matmul_precision("high")

<<<<<<< HEAD
    datamodule = ParrotDataModule(f"{tsv_file}",
                                num_classes=num_classes,
                                datatype=f"{datatype}", 
                                split_file=f"{split_file}",
                                ignore_warnings=ignore_warnings,
                                batch_size=batch_size)

    storage = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner()

    sampler = optuna.samplers.TPESampler(n_startup_trials=50)
=======
datamodule = ParrotDataModule("meta.tsv",
                              num_classes=1,
                              datatype="residues", 
                              split_file="meta_2023_06_28_split_file.txt",
                              ignore_warnings=True,
                              batch_size=256)

# this can improve performance for tensor cores cards - should figure out how to do this more elegantly.
torch.set_float32_matmul_precision("high")

study_name = "metapredict_layer_norm_b256"
storage = f"sqlite:///{study_name}.db"

pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

# quasi-random search is recommended by google because its more efficient at search than random
# but isnt pigeon holed by adaptive sampling approach like TPE (e.g., if you want to rerank based
# on another metric you can)

# stage 1 of two stage optimization
#sampler = optuna.samplers.QMCSampler(scramble=True,seed=42)
#
#study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage, 
#                            direction='minimize', pruner=pruner,load_if_exists=True)
#
#study.optimize(lambda trial: objective(trial, datamodule), n_trials=350)

# stage 2 of two stage optimization
sampler = optuna.samplers.TPESampler(n_startup_trials=0)

study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage, 
                            direction='minimize', pruner=pruner,load_if_exists=True)
study.optimize(lambda trial: objective(trial, datamodule), n_trials=10000)

>>>>>>> 1c823d06dfc6fbcfe96f753a39f2a173f4bf828b

    study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage, 
                                direction='minimize', pruner=pruner, load_if_exists=True)

<<<<<<< HEAD
    study.optimize(lambda trial: objective(trial, datamodule, config), n_trials=n_trials)

def parse_and_write_args_to_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_name', default='parrot_study', help='Name of the study')
    parser.add_argument('--tsv_file', default=None, help='Path to the tsv file')
    parser.add_argument('--split_file', default=None, help='Path to the split file')
    

    parser.add_argument('--config', default=None, help='Path to the configuration file')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU device ID to use')
    parser.add_argument('--optimizer_name', help='Optimizer to use')
    parser.add_argument('--num_lstm_layers',nargs=2, type=int, help=f'The number of lstm layers to consider.'
                                                            'The first index will be used as the min.' 
                                                            'The second will be used as the max')
    parser.add_argument('--lstm_hidden_size',nargs=2, type=int, help=f'The number of hidden units in the lstm layers.')

    parser.add_argument('--dropout', nargs=2, type=float, help='The range of dropout rates on dense linear layers to test'
                                                                'The first index will be used as the min.' 
                                                                'The second will be used as the max')
    
    parser.add_argument('--eps', nargs=2, type=float, help='The range of eps values to add to denominator of adam optimizer'
                                                                'The first index will be used as the min.' 
                                                                'The second will be used as the max')
    
    parser.add_argument('--learn_rate', nargs=2, type=float, help='The learning rate of the optimizer'
                                                                'The first index will be used as the min.' 
                                                                'The second will be used as the max')    

    args = parser.parse_args()

    if args.config is None:
        # Set the default configuration file path to the data folder of the package
        default_config_path = os.path.join(get_directory(), 'config.yaml')
        args.config = default_config_path

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    for arg_name, arg_value in vars(args).items():
        if arg_name == 'config' or arg_value is None:
            continue
        
        # Update the corresponding key in the configuration dictionary
        if isinstance(arg_value, list):
            # Handle list values
            config[arg_name].update({"min":arg_value[0]})
            config[arg_name].update({"max":arg_value[1]})
            if len(arg_value) > 2:
                assert arg_value[2] in [True, False], f'log value must be True or False, not {arg_value[2]}'
                config[arg_name].update({"log":arg_value[2]})
        else:
            # Handle single values
            config[arg_name] = arg_value

    args.config = 'parameter_sweep.yaml'

    with open('parameter_sweep.yaml', 'w') as config_file:
        final_config['yaml'] = args.config
        yaml.safe_dump(config, config_file)

    # Set defaults from the updated config dictionary
    parser.set_defaults(**final_config)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_and_write_args_to_yaml()
    print(args.config)
    with open(args.config) as config_file:
        final_config = yaml.safe_load(config_file)
    print(final_config)
    run_optimization(final_config)

    
=======
>>>>>>> 1c823d06dfc6fbcfe96f753a39f2a173f4bf828b
