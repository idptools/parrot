#!/usr/bin/env python
import argparse
import os

import optuna
import pytorch_lightning as pl
import torch
import wandb
from optuna.integration import PyTorchLightningPruningCallback

# from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from parrot.brnn_architecture import BRNN_PARROT, ParrotDataModule

# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor,
# from lightning.pytorch.loggers import WandbLogger


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


def objective(trial: optuna.trial.Trial, datamodule: pl.LightningDataModule, config):
    """Objective function for Optuna to optimize.

    Parameters
    ----------
    trial : optuna.trial.Trial
        optuna trial object used for optimizing hyperparameters
    datamodule : pl.LightningDataModule
        a ParrotDataModule, which is a Lightning datamodule object, for the machine learning task.
    config : dict
        Configuration dictionary for the hyperparameter search spaces

    Returns
    -------
    float
        Metric being monitored for the model at the epoch.
    """

    datatype = datamodule.datatype
    num_classes = datamodule.num_classes
    problem_type = datamodule.problem_type
    input_size = datamodule.input_size
    batch_size = datamodule.batch_size

    monitor = config["monitor"]
    min_delta = config["min_delta"]
    direction = config["direction"]
    min_epoch = config["min_epochs"]
    max_epoch = config["max_epochs"]

    # Define the hyperparameter search space using trial.suggest_*
    hparams = {
        "optimizer_name": trial.suggest_categorical(
            "optimizer_name", config["optimizer_choices"]
        ),
        "num_lstm_layers": trial.suggest_int(
            "num_lstm_layers",
            config["num_lstm_layers_min"],
            config["num_lstm_layers_max"],
        ),
        "lstm_hidden_size": trial.suggest_int(
            "lstm_hidden_size",
            config["lstm_hidden_size_min"],
            config["lstm_hidden_size_max"],
        ),
        "learn_rate": trial.suggest_float(
            "learn_rate",
            config["learn_rate_min"],
            config["learn_rate_max"],
            log=config["learn_rate_log"],
        ),
        "gradient_clip_val": trial.suggest_float(
            "gradient_clip_val",
            config["gradient_clip_val_min"],
            config["gradient_clip_val_max"],
        ),
        "input_size": input_size,
        "num_classes": num_classes,
        "problem_type": problem_type,
        "datatype": datatype,
        "batch_size": batch_size,
        "min_epoch": min_epoch,
        "max_epoch": max_epoch,
        "distributed": config["distributed"],
    }

    num_linear_layers = trial.suggest_int(
        "num_linear_layers",
        config["num_linear_layers_min"],
        config["num_linear_layers_max"],
    )

    hparams["num_linear_layers"] = num_linear_layers
    hparams["direction"] = direction
    hparams["monitor"] = monitor

    # Always suggest linear_hidden_size, even if num_linear_layers == 1
    # When num_linear_layers == 1, this parameter will be ignored by the architecture
    # but it's better to always include it for consistent hyperparameter exploration
    hparams["linear_hidden_size"] = trial.suggest_int(
        "linear_hidden_size",
        config["linear_hidden_size_min"],
        config["linear_hidden_size_max"],
    )

    # Always suggest dropout for linear layers (when num_linear_layers > 1, dropout is applied)
    hparams["dropout"] = trial.suggest_float(
        "dropout",
        config["dropout_min"],
        config["dropout_max"],
    )

    if hparams["optimizer_name"] == "SGD":
        hparams["momentum"] = trial.suggest_float(
            "momentum",
            config["momentum_min"],
            config["momentum_max"],
        )

    elif hparams["optimizer_name"] == "AdamW":
        hparams["beta1"] = trial.suggest_float(
            "beta1", config["beta1_min"], config["beta1_max"]
        )

        hparams["beta2"] = trial.suggest_float(
            "beta2", config["beta2_min"], config["beta2_max"]
        )

        hparams["eps"] = trial.suggest_float(
            "eps", config["eps_min"], config["eps_max"], log=config["eps_log"]
        )

        hparams["weight_decay"] = trial.suggest_float(
            "weight_decay",
            config["weight_decay_min"],
            config["weight_decay_max"],
            log=config["weight_decay_log"],
        )

    elif hparams["optimizer_name"] == "Adam":
        hparams["beta1"] = trial.suggest_float(
            "beta1", config["beta1_min"], config["beta1_max"]
        )

        hparams["beta2"] = trial.suggest_float(
            "beta2", config["beta2_min"], config["beta2_max"]
        )

        hparams["eps"] = trial.suggest_float(
            "eps", config["eps_min"], config["eps_max"], log=config["eps_log"]
        )

        hparams["weight_decay"] = trial.suggest_float(
            "weight_decay",
            config["weight_decay_min"],
            config["weight_decay_max"],
            log=config["weight_decay_log"],
        )

    else:
        hparams["momentum"] = None

    # Create the model with the suggested hyperparameters
    model= BRNN_PARROT(**hparams)

    for hparam_name, hparam_value in hparams.items():
        print(f"{hparam_name} = ", hparam_value)

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=10,
        verbose=False,
        mode=direction[:3],
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # pruning_callback = PyTorchLightningPruningCallback(trial, monitor=monitor)

    wandb_logger = WandbLogger(
        name=f"run{trial.number}", project=f"{config['study_name']}"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        filename="epoch{epoch:03d}_val_loss{epoch_val_loss:.2f}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    wandb_logger.watch(model)

    # Determine accelerator and device configuration
    if torch.cuda.is_available() and not config["force_cpu"]:
        accelerator = "gpu"
        devices = config["gpu_id"]
    else:
        accelerator = "cpu"
        devices = 1  # CPU accelerator expects an integer

    trainer = pl.Trainer(
        gradient_clip_val=hparams["gradient_clip_val"],
        precision="16-mixed",
        logger=wandb_logger,
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
        accelerator=accelerator,
        devices=devices,
        # callbacks = [pruning_callback, early_stop_callback, checkpoint_callback, lr_monitor],
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
    )

    trainer.logger.log_hyperparams(hparams)

    trainer.fit(model, datamodule=datamodule)

    wandb_logger.experiment.unwatch(model)
    wandb.finish()

    # Return the validation loss as the objective value for Optuna
    return trainer.callback_metrics[monitor].detach()


def run_optimization(
    config,
    study_name,
    tsv_file,
    split_file,
    num_classes,
    datatype,
    batch_size,
    ignore_warnings=False,
):
    """Runs the optimization using Optuna.

    Parameters
    ----------
    config : dict
        Dictionary containing the hyperparameter search space configuration.
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

    n_trials = config["n_trials"]
    direction = config["direction"]
    warm_up_trials = config["warm_up_trials"]
    
    # this can improve performance for tensor cores cards
    if determine_matmul_precision():
        torch.set_float32_matmul_precision("high")

    # Set default num_workers if not specified
    if config.get("num_workers") is None:
        config["num_workers"] = (
            os.cpu_count() if os.cpu_count() <= 32 else os.cpu_count() // 4
        )

    datamodule = ParrotDataModule(
        f"{tsv_file}",
        num_classes=num_classes,
        datatype=f"{datatype}",
        split_file=f"{split_file}",
        ignore_warnings=ignore_warnings,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        distributed=config["distributed"]
    )

    storage = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=warm_up_trials)

    sampler = optuna.samplers.TPESampler(n_startup_trials=warm_up_trials)

    study = optuna.create_study(
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        direction=f"{direction}",
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, datamodule, config), n_trials=n_trials
    )


def parse_args():
    """
    Parse command line arguments for optimization. Provides sensible defaults for all parameters.

    Returns
    -------
    dict
        Dictionary containing the configuration parameters.
    """
    parser = argparse.ArgumentParser(
        description="PARROT hyperparameter optimization without YAML config files"
    )
    
    # Config file argument
    parser.add_argument(
        "--config", default=None, help="Path to the configuration file (optional)."
    )
    
    # Required arguments with defaults
    parser.add_argument(
        "--num_classes",
        default=1,
        type=int,
        help="Number of classes. For regression, this value must be set to 1.",
    )
    parser.add_argument(
        "--datatype", 
        default="sequence", 
        help="Type of data. Must be 'sequence' or 'residues'"
    )
    parser.add_argument(
        "--tsv_file", 
        default=None,
        help="Path to the training tsv file."
    )
    parser.add_argument(
        "--split_file",
        default=None,
        help="Path to the file indicating the train/validation/test split.",
    )
    parser.add_argument(
        "--study_name", 
        default="parrot_optimization", 
        help="Name of the study. Used for WandB logging."
    )
    parser.add_argument(
        "--batch_size", 
        default=32, 
        type=int, 
        help="The batch size of the model."
    )
    parser.add_argument(
        "--ignore_warnings",
        action="store_true",
        help="Optionally ignore parrot warnings.",
    )
    parser.add_argument(
        "--gpu_id", 
        nargs="+", 
        default=[0], 
        type=int, 
        help="GPU device ID(s) to use."
    )

    # Optimization parameters
    parser.add_argument(
        "--n_trials",
        default=100,
        type=int,
        help="Number of optimization trials to run.",
    )
    parser.add_argument(
        "--direction",
        default="minimize",
        choices=["minimize", "maximize"],
        help="Direction of optimization.",
    )
    parser.add_argument(
        "--warm_up_trials",
        default=10,
        type=int,
        help="Number of warm-up trials for pruning.",
    )

    # Hyperparameter ranges
    parser.add_argument(
        "--optimizer_name",
        nargs="+",
        default=["AdamW"],
        help="List of optimizers to potentially use. Currently supported are adamw and sgd",
    )
    parser.add_argument(
        "--learn_rate",
        nargs=2,
        type=float,
        default=[1e-5, 1e-2],
        help="Learning rate range [min, max]",
    )
    parser.add_argument(
        "--num_lstm_layers",
        nargs=2,
        type=int,
        default=[1, 3],
        help="LSTM layers range [min, max]",
    )
    parser.add_argument(
        "--lstm_hidden_size",
        nargs=2,
        type=int,
        default=[32, 256],
        help="LSTM hidden size range [min, max]",
    )
    parser.add_argument(
        "--num_linear_layers",
        nargs=2,
        type=int,
        default=[1, 4],
        help="Linear layers range [min, max]",
    )
    parser.add_argument(
        "--linear_hidden_size",
        nargs=2,
        type=int,
        default=[32, 512],
        help="Linear hidden size range [min, max]",
    )
    parser.add_argument(
        "--dropout",
        nargs=2,
        type=float,
        default=[0.0, 0.5],
        help="Dropout range [min, max]",
    )

    # SGD specific parameters
    parser.add_argument(
        "--momentum",
        nargs=2,
        type=float,
        default=[0.8, 0.99],
        help="SGD momentum range [min, max]",
    )

    # AdamW specific parameters
    parser.add_argument(
        "--beta1",
        nargs=2,
        type=float,
        default=[0.85, 0.95],
        help="AdamW beta1 range [min, max]",
    )
    parser.add_argument(
        "--beta2",
        nargs=2,
        type=float,
        default=[0.98, 0.9999],
        help="AdamW beta2 range [min, max]",
    )
    parser.add_argument(
        "--eps",
        nargs=2,
        type=float,
        default=[1e-9, 1e-7],
        help="AdamW eps range [min, max]",
    )
    parser.add_argument(
        "--weight_decay",
        nargs=2,
        type=float,
        default=[1e-6, 1e-2],
        help="AdamW weight decay range [min, max]",
    )

    # Additional parameters
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loader workers (auto-detected if not specified)",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument(
        "--gradient_clip_val",
        nargs=2,
        type=float,
        default=[0.5, 2.0],
        help="Gradient clipping value range [min, max]",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="epoch_val_loss",
        help="Metric to monitor for optimization (supports both epoch_val_loss and val_loss formats)",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum change in monitored metric for early stopping",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=5,
        help="Minimum number of epochs to train",
    )
    parser.add_argument(
        "--optimize_max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs for optimization trials",
    )

    args = parser.parse_args()
    
    # Start with empty config, then load from file if provided
    config = {}
    
    # If config file is provided, load it first
    if args.config is not None:
        try:
            import yaml
            with open(args.config) as config_file:
                yaml_config = yaml.safe_load(config_file)
            # Handle case where YAML file is empty or None
            if yaml_config is not None:
                # Extract values from YAML structure if it uses the nested format
                for key, value in yaml_config.items():
                    if isinstance(value, dict) and "value" in value:
                        config[key] = value["value"]
                    else:
                        config[key] = value
        except FileNotFoundError:
            print(f"Warning: Config file {args.config} not found. Using command line arguments and defaults.")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Override with command line arguments (only if explicitly provided)
    # Store defaults to check if values were explicitly set
    parser_defaults = {
        'num_classes': 1,
        'datatype': 'sequence', 
        'tsv_file': None,
        'split_file': None,
        'study_name': 'parrot_optimization',
        'batch_size': 32,
        'ignore_warnings': False,
        'gpu_id': [0],
        'n_trials': 100,
        'direction': 'minimize',
        'warm_up_trials': 10,
        'optimizer_name': ['AdamW'],
        'learn_rate': [1e-5, 1e-2],
        'num_lstm_layers': [1, 3],
        'lstm_hidden_size': [32, 256],
        'num_linear_layers': [1, 4],
        'linear_hidden_size': [32, 512],
        'dropout': [0.0, 0.5],
        'momentum': [0.8, 0.99],
        'beta1': [0.85, 0.95],
        'beta2': [0.98, 0.9999],
        'eps': [1e-9, 1e-7],
        'weight_decay': [1e-6, 1e-2],
        'gradient_clip_val': [0.5, 2.0],
        'monitor': 'epoch_val_loss',
        'min_delta': 0.001,
        'min_epochs': 5,
        'max_epochs': 100,
        'optimize_max_epochs': 100,
        'distributed': False,
        'num_workers': None,
        'force_cpu': False
    }
    
    for arg_name, arg_value in vars(args).items():
        if arg_name != "config":
            # Only override config if the argument was explicitly provided (different from default)
            # or if the argument is not in the config file
            if (arg_name not in parser_defaults or 
                arg_value != parser_defaults[arg_name] or 
                arg_name not in config):
                config[arg_name] = arg_value
    
    # Validate that required arguments are present
    if "tsv_file" not in config or config["tsv_file"] is None:
        parser.error("--tsv_file is required either as a command line argument or in the config file")
    
    # Convert monitor metric from new format (epoch_val_loss) to old format (val_loss) for compatibility
    monitor_metric = config.get("monitor", "epoch_val_loss")
    if monitor_metric.startswith("epoch_"):
        converted_monitor = monitor_metric.replace("epoch_", "")
    else:
        converted_monitor = monitor_metric
    
    # Convert args to a config dictionary with the structure expected by the optimization functions
    final_config = {
        "num_classes": config.get("num_classes", 1),
        "datatype": config.get("datatype", "sequence"),
        "tsv_file": config["tsv_file"],
        "split_file": config.get("split_file", None),
        "study_name": config.get("study_name", "parrot_optimization"),
        "batch_size": config.get("batch_size", 32),
        "ignore_warnings": config.get("ignore_warnings", False),
        "gpu_id": config.get("gpu_id", [0]),
        "n_trials": config.get("n_trials", 100),
        "direction": config.get("direction", "minimize"),
        "warm_up_trials": config.get("warm_up_trials", 10),
        "distributed": config.get("distributed", False),
        "num_workers": config.get("num_workers", None),
        "force_cpu": config.get("force_cpu", False),
        
        # Set parameters needed by objective function that weren't in the original config structure
        "monitor": converted_monitor,
        "min_delta": config.get("min_delta", 0.001),
        "min_epochs": config.get("min_epochs", 5),
        "max_epochs": config.get("optimize_max_epochs", config.get("max_epochs", 100)),
        
        # Convert optimizer choices to the format expected by the objective function
        "optimizer_choices": config.get("optimizer_name", ["AdamW"]),
        
        # Convert hyperparameter ranges to the format expected by the objective function
        "learn_rate_min": config.get("learn_rate", [1e-5, 1e-2])[0],
        "learn_rate_max": config.get("learn_rate", [1e-5, 1e-2])[1], 
        "learn_rate_log": True,
        "num_lstm_layers_min": config.get("num_lstm_layers", [1, 3])[0],
        "num_lstm_layers_max": config.get("num_lstm_layers", [1, 3])[1],
        "lstm_hidden_size_min": config.get("lstm_hidden_size", [32, 256])[0],
        "lstm_hidden_size_max": config.get("lstm_hidden_size", [32, 256])[1],
        "num_linear_layers_min": config.get("num_linear_layers", [1, 4])[0],
        "num_linear_layers_max": config.get("num_linear_layers", [1, 4])[1],
        "linear_hidden_size_min": config.get("linear_hidden_size", [32, 512])[0],
        "linear_hidden_size_max": config.get("linear_hidden_size", [32, 512])[1],
        "dropout_min": config.get("dropout", [0.0, 0.5])[0],
        "dropout_max": config.get("dropout", [0.0, 0.5])[1],
        "momentum_min": config.get("momentum", [0.8, 0.99])[0],
        "momentum_max": config.get("momentum", [0.8, 0.99])[1],
        "beta1_min": config.get("beta1", [0.85, 0.95])[0],
        "beta1_max": config.get("beta1", [0.85, 0.95])[1],
        "beta2_min": config.get("beta2", [0.98, 0.9999])[0],
        "beta2_max": config.get("beta2", [0.98, 0.9999])[1],
        "eps_min": config.get("eps", [1e-9, 1e-7])[0],
        "eps_max": config.get("eps", [1e-9, 1e-7])[1],
        "eps_log": True,
        "weight_decay_min": config.get("weight_decay", [1e-6, 1e-2])[0],
        "weight_decay_max": config.get("weight_decay", [1e-6, 1e-2])[1],
        "weight_decay_log": True,
        "gradient_clip_val_min": config.get("gradient_clip_val", [0.5, 2.0])[0],
        "gradient_clip_val_max": config.get("gradient_clip_val", [0.5, 2.0])[1],
    }
    
    return final_config


if __name__ == "__main__":
    config = parse_args()
    
    run_optimization(
        config,
        config["study_name"],
        config["tsv_file"],
        config["split_file"],
        config["num_classes"],
        config["datatype"],
        config["batch_size"],
        ignore_warnings=config["ignore_warnings"],
    )
