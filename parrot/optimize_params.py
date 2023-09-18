import argparse
import yaml
import os
from parrot import get_directory
import datetime

import torch
from parrot.brnn_architecture import BRNN_MtM, BRNN_MtO, ParrotDataModule
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import pytorch_lightning as pl

def determine_matmul_precision():
    """returns True if the GPU supports Tensor Core matmul operations, False otherwise

    Returns
    -------
    bool
        A boolean indicating whether the GPU supports Tensor Core matmul operations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return torch.cuda.get_device_properties(device).major >= 7
    else: 
        return False
    
def objective(trial : optuna.trial.Trial, datamodule : pl.LightningDataModule, config):
    """Objective function for Optuna to optimize.

    Parameters
    ----------
    trial : optuna.trial.Trial
        optuna trial object used for optimizing hyperparameters
    datamodule : pl.LightningDataModule
        a ParrotDataModule, which is a Lightning datamodule object, for the machine learning task.
    config : _type_
        Yaml configuration file for the hyperparameter search spaces

    Returns
    -------
    float
        validation loss for the model at the epoch.
    """

    datatype = datamodule.datatype
    num_classes = datamodule.num_classes
    problem_type = datamodule.problem_type
    input_size = datamodule.input_size
    batch_size = datamodule.batch_size
    
    # Define the hyperparameter search space using trial.suggest_*
    hparams = {
        config['optimizer_name']['name']: trial.suggest_categorical(config['optimizer_name']['name'],
                                                               list(config['optimizer_name']['choices'].values())
                                                               ), 
        
        config['num_lstm_layers']['name'] : trial.suggest_int(config['num_lstm_layers']['name'],
                                                         config['num_lstm_layers']['min'],
                                                         config['num_lstm_layers']['max']),

        config['lstm_hidden_size']['name'] : trial.suggest_int(config['lstm_hidden_size']['name'],
                                                           config['lstm_hidden_size']['min'],
                                                            config['lstm_hidden_size']['max']),

        config['learn_rate']['name'] : trial.suggest_float(config['learn_rate']['name'],
                                                      config['learn_rate']['min'],
                                                      config['learn_rate']['max'],
                                                      log=config['learn_rate']['log']
                                                      ),
        'input_size': input_size,
        'num_classes': num_classes,
        'problem_type': problem_type,
        'datatype': datatype,
        'batch_size': batch_size,
    }

    num_linear_layers = trial.suggest_int(config['num_linear_layers']['name'],
                                config['num_linear_layers']['min'],
                                config['num_linear_layers']['max'])
    
    hparams[config['num_linear_layers']['name']] = num_linear_layers
    
    if num_linear_layers > 1:    
        hparams[config['linear_hidden_size']['name']] = trial.suggest_int(config['linear_hidden_size']['name'],
                                                                        config['linear_hidden_size']['min'],
                                                                        config['linear_hidden_size']['max'])
        
        hparams['use_dropout'] = trial.suggest_categorical(name="use_dropout", choices=[True, False]) 

        if hparams['use_dropout']:
           hparams[config['dropout']['name']] = trial.suggest_float(config['dropout']['name'],
                                                            config['dropout']['min'],
                                                            config['dropout']['max'])
        else:
            hparams[config['dropout']['name']] = 0.0
            
    
    if hparams['optimizer_name'] == 'SGD':
        # shrug - could probably tune gradient clip, but helped and worked so good enough
        
        hparams['momentum'] = trial.suggest_float(config['momentum']['name'],
                                                config['momentum']['min'],
                                                config['momentum']['max'])
        gradient_clip_val = 1.0
    elif hparams['optimizer_name'] == 'AdamW':
        hparams['beta1'] = trial.suggest_float(config['beta1']['name'],       
                                                 config['beta1']['min'],                                       
                                                config['beta1']['max'])       
        
        hparams['beta2'] = trial.suggest_float(config['beta2']['name'],      
                                                 config['beta2']['min'],  
                                                config['beta2']['max'])   
        
        hparams['eps'] = trial.suggest_float(config['eps']['name'],
                                                config['eps']['min'],
                                                config['eps']['max'])
             
        hparams['weight_decay'] = trial.suggest_float(config['weight_decay']['name'],
                                        config['weight_decay']['min'],
                                        config['weight_decay']['max'])
        gradient_clip_val = None
    elif hparams['optimizer_name'] == 'Adam':
        hparams['beta1'] = trial.suggest_float(config['beta1']['name'],       
                                                 config['beta1']['min'],                                       
                                                config['beta1']['max'])       
        
        hparams['beta2'] = trial.suggest_float(config['beta2']['name'],      
                                                 config['beta2']['min'],  
                                                config['beta2']['max'])   
        
        hparams['eps'] = trial.suggest_float(config['eps']['name'],
                                                config['eps']['min'],
                                                config['eps']['max'])
             
        hparams['weight_decay'] = trial.suggest_float(config['weight_decay']['name'],
                                        config['weight_decay']['min'],
                                        config['weight_decay']['max'])
        gradient_clip_val = None
    else:
        hparams['momentum'] = None
        gradient_clip_val = None

    for hparam_name, hparam_value in hparams.items():
        print(f"{hparam_name} = ", hparam_value)

    if datatype == 'sequence':
        model = BRNN_MtO(**hparams)
    elif datatype == 'residues':
        model = BRNN_MtM(**hparams)

    early_stop_callback = EarlyStopping(
                                monitor='epoch_val_loss',
                                min_delta=0.000,
                                patience=5,
                                verbose=False,
                                mode='min'
                                )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="epoch_val_loss")
    
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    wandb_logger = WandbLogger(name=f"run{trial.number}",
                               project=f"{config['study_name']['value']}")
    

    checkpoint_callback = ModelCheckpoint(
                            monitor='epoch_val_loss',
                            filename="epoch{epoch:03d}_val_loss{epoch_val_loss:.2f}",
                            auto_insert_metric_name=False,
                            save_on_train_epoch_end=False,
    )
    wandb_logger.watch(model)

    trainer = pl.Trainer(
        gradient_clip_val = gradient_clip_val,
        precision = "16-mixed",  
        logger = wandb_logger,
        min_epochs = 100,
        max_epochs = 250,
        accelerator = "auto",
        devices = config['gpu_id'],
        callbacks = [pruning_callback, early_stop_callback, checkpoint_callback, swa_callback],
    )

    trainer.logger.log_hyperparams(hparams)

    trainer.fit(model, datamodule=datamodule)
    
    wandb_logger.experiment.unwatch(model)
    wandb.finish()

    # Return the validation loss as the objective value for Optuna
    return trainer.callback_metrics['epoch_val_loss'].detach()

def run_optimization(config, study_name, tsv_file, split_file, num_classes,
                     datatype,batch_size, ignore_warnings=False):
    """Runs the optimization using Optuna.

    Parameters
    ----------
    config : dict
        Dictionary containing the hyperparameter search space from the config yaml file.
    study_name : str
        The name of the optuna study. This is what is used for logging to WandB.
    tsv_file : str
        The path to the training/validation/test split tsv file.
    split_file : str
        The path to the file indicating the train/validation/test splits
    num_classes : int
        Number of classes for the machine learning task, use 1 for regression.
    datatype : str
        Must be one of either residues or sequence.
    batch_size : int
        batch size to train your model with.
    ignore_warnings : bool, optional
        Ignore parrot warnings, by default False
    """

    n_trials = config['n_trials']['value']
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

    sampler = optuna.samplers.TPESampler(n_startup_trials=20)

    study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage, 
                                direction='minimize', pruner=pruner, load_if_exists=True)

    study.optimize(lambda trial: objective(trial, datamodule, config), n_trials=n_trials)

def parse_and_write_args_to_yaml():
    """Provide optional CLI overwrites to default yaml configuration file.
    The default yaml configuration file is located in the parrot package directory under data/
    Hyperparameter sweep search spaces are output to a yaml file in the current working directory.

    Returns
    -------
    argparse.Namespace
        argparse object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--config', default=None, help='Path to the configuration file.')
    parser.add_argument('--num_classes', default=None, type=int, help='Number of classes. For regression, this value must be set to 1.')
    parser.add_argument('--datatype', default=None, help="Type of data. Must be 'sequence' or 'residues")
    parser.add_argument('--tsv_file', default=None, help='Path to the training tsv file.')

    parser.add_argument('--split_file', default=None, help='Path to the file indicating the train/validation/test split.')
    parser.add_argument('--study_name', default=None, help='Name of the study. Used for WandB logging.')
    parser.add_argument('--batch_size', default=None, type=int, help='The batch size of the model.')
    parser.add_argument('--ignore_warnings', default=None, type=bool, help='Optionally ignore parrot warnings.')
    parser.add_argument('--gpu_id', nargs='+', default=[0], type=int, help='GPU device ID(s) to use.')
    
    # optional overrides to default configs
    parser.add_argument('--optimizer_name', nargs='+', help='List of optimizers to potentially use. Currently support are AdamW and SGD+momentum')
    
    parser.add_argument('--learn_rate', nargs=2, type=float, help='The learning rate of the optimizer. '
                                                                'The first index will be used as the min. ' 
                                                                'The second will be used as the max.')    
    


    parser.add_argument('--num_lstm_layers', nargs=2, type=int, help=f'The number of lstm layers to consider. '
                                                            'The first index will be used as the min. ' 
                                                            'The second will be used as the max.')

    parser.add_argument('--lstm_hidden_size', nargs=2, type=int, help=f'The number of hidden units in the lstm layers.')

    parser.add_argument('--num_linear_layers', nargs=2, type=int, help=f'The number of linear layers to consider. '
                                                            'The first index will be used as the min. ' 
                                                            'The second will be used as the max.')
    
    parser.add_argument('--linear_hidden_size', nargs=2, type=int, help=f'The number of hidden units in the linear layers.'
                                                            'The first index will be used as the min. ' 
                                                            'The second will be used as the max.')
    
    parser.add_argument('--dropout', nargs=2, type=float, help='The range of dropout rates on dense linear layers to test. '
                                                                'The first index will be used as the min. ' 
                                                                'The second will be used as the max.')
    
    # SGD specific parameters
    parser.add_argument('--momentum', nargs=2, type=float, help='Parameter only used if using an SGD optimizer.'
                                                                'The range of momentum values to add to denominator of SGD optimizer. '
                                                                'The first index will be used as the min. '
                                                                'The second will be used as the max.')

    # AdamW specific parameters
    parser.add_argument('--beta1', nargs=2, type=float, help='Parameter only used if using an Adam-based optimizer. '
                                                                'These will define range of beta1 values to add to denominator of adamW optimizer. '
                                                                'The first index will be used as the min. ' 
                                                                'The second will be used as the max.')
    
    parser.add_argument('--beta2', nargs=2, type=float, help='Parameter only used if using an Adam-based optimizer. '
                                                                'The range of beta2 values to add to denominator of adamW optimizer. '
                                                                'The first index will be used as the min. ' 
                                                                'The second will be used as the max.')
    
    parser.add_argument('--eps', nargs=2, type=float, help='Parameter only used if using an Adam-based optimizer. '
                                                                'The range of eps values to add to denominator of adamW optimizer. '
                                                                'The first index will be used as the min.' 
                                                                'The second will be used as the max.')
    
    parser.add_argument('--weight_decay', nargs=2, type=float, help='Parameter only used if using an Adam-based optimizer. '
                                                                'The range of weight decay values to add to denominator of adamW optimizer. '
                                                                'The first index will be used as the min.' 
                                                                'The second will be used as the max.')

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
        if isinstance(arg_value, list) and arg_name != 'gpu_id':
            # Handle list values
            config[arg_name].update({"min" : arg_value[0]})
            config[arg_name].update({"max" : arg_value[1]})
            if len(arg_value) > 2:
                assert arg_value[2] in [True, False], f'log value must be True or False, not {arg_value[2]}'
                config[arg_name].update({"log":arg_value[2]})
        else:
            # Handle single values or list gpu_id (? check this)
            config[arg_name] = arg_value


    args.config = f'{config["study_name"]["value"]}_param_sweep_{datetime.date.today().strftime("%Y_%m_%d")}.yaml'
    
    with open(args.config, 'w') as config_file:
        yaml.safe_dump(config, config_file)

    # Set defaults from the updated config dictionary 
    # is this the best spot for this? parsing code feels a bit odd but works    
    parser.set_defaults(**config)

    return args

if __name__ == "__main__":
    args = parse_and_write_args_to_yaml()
    with open(args.config) as config_file:
        final_config = yaml.safe_load(config_file)

    run_optimization(final_config,
                final_config['study_name']['value'],
                final_config['tsv_file']['value'], 
                final_config['split_file']['value'], 
                final_config['num_classes']['value'],
                final_config['datatype']['value'], 
                final_config['batch_size']['value'],
                ignore_warnings=final_config['ignore_warnings']['value'])