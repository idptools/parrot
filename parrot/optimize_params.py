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
        hparams['momentum'] = trial.suggest_float('momentum', 0.8, 1.0)
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
    print("lstm_hidden_size",hparams.get('lstm_hidden_size'))
    print("num_lstm_layers",hparams.get('num_lstm_layers'))
    print("linear_hidden_size",hparams.get('linear_hidden_size'),0)
    print("num_linear_layers",hparams.get('num_linear_layers'),1)
    print("dropout",hparams.get('dropout'))
    print("beta1",hparams.get('beta1', 0.9))
    print("beta2",hparams.get('beta2', 0.999))
    print("eps",hparams.get('eps', 1e-8))

    if datatype == 'sequence':
        model = BRNN_MtO(**hparams)
    elif datatype == 'residues':
        model = BRNN_MtM(**hparams)

    early_stop_callback = EarlyStopping(
                                monitor='average_epoch_val_loss',
                                min_delta=0.000,
                                patience=5,
                                verbose=False,
                                mode='min'
                                )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="average_epoch_val_loss")
    
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    wandb_logger = WandbLogger(name=f"run{trial.number}",
                               project='metapredict_b64_mlp')
    
    wandb_logger.watch(model)

    # gradient_clip_val = 1.0
    trainer = pl.Trainer(
        gradient_clip_val=gradient_clip_val,
        precision="16-mixed",  
        logger=wandb_logger,
        enable_checkpointing=True,
        # limit_train_batches=5,
        max_epochs=50,
        accelerator="auto",
        devices=[int(config.gpu_id)],
        callbacks = [pruning_callback,
                     early_stop_callback],
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

    datamodule = ParrotDataModule(f"{tsv_file}",
                                num_classes=num_classes,
                                datatype=f"{datatype}", 
                                split_file=f"{split_file}",
                                ignore_warnings=ignore_warnings,
                                batch_size=batch_size)

    storage = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner()

    sampler = optuna.samplers.TPESampler(n_startup_trials=50)

    study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage, 
                                direction='minimize', pruner=pruner, load_if_exists=True)

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

    