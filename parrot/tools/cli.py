import random

import numpy as np
import torch

from parrot.package_version import PARROT_VERSION


def set_random_seed(seed):
    """Seed every random number generator PARROT draws from

    PARROT uses Python's `random` module (class balancing), NumPy (train/val/test
    splits and cross-validation folds) and PyTorch (network initialisation and
    batch shuffling). Seeding all three makes splits and training runs
    reproducible on the same machine. On NVIDIA GPUs, cuDNN is also told to use
    only deterministic algorithms: some of its default convolution algorithms
    are not, which made seeded U-Net training differ between runs (at no
    measurable cost in speed on an RTX A4000).

    Parameters
    ----------
    seed : int or None
        The seed to use. If None, nothing is seeded and runs are not reproducible.

    Returns
    -------
    None
    """

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device and the MPS (Apple-silicon) device
    torch.backends.cudnn.deterministic = True


def print_startup(program_name, silent, print_parrot=False):
    """
    Function that prints a welcoming startup message with a 
    snazzy picture of a parrot.
    
    Parameters
    -----------
    program_name : str
        Name of the cli program being executed

    print_parrot : bool
        If set to True, a snazzy parrot pictures is printed

    silent : bool
        Boolean which, if set to true, means no message is printed 

    Returns
    --------
    None
        No return but will print a startup message unless silent=True

    
    """

    # if silent print nothing
    if silent:
        return

    if print_parrot is True:
        print(r"...................................")
        print(r"|                                 |")
        print(r"|    _        Welcome to PARROT!  |")
        print(r"|  /` '\                          |")    
        print(r"|/| @   l                         |")
        print(r"|\|      \                        |")
        print(r"|  `\     `\_                     |")
        print(r"|    \    __ `\                   |")
        print(r"|    l  \   `\ `\__               |")
        print(r"|     \  `\./`     ``\            |")
        print(r"|       \ ____ / \   l            |")
        print(r"|         ||  ||  )  /            |")
        print(r"|-------(((-(((---l /-------------|")
        print(r"|                l /              |")
        print(r"|               / /               |")
        print(r"|              / /                |")
        print(r"|             //                  |")
        print(r"|            /                    |")
        print(r"...................................")

    print(f'\n{program_name} (version {PARROT_VERSION})')
    print("...................................")
    print(f'Developed by Dan Griffith\n   Holehouse lab, WUSTL')
    print("")
    print("")
    


def print_settings(silent, buffer_length=14, **kwargs):
    """
    Function that dynamically prints out a list of key-value pairs
    as passed. Means that when this function can be 
    


    """


    # ............................................................
    def strbuffer(local_string):
        """
        Short internal function that pads a string so it is 
        buffer_length long. Note BUFFER_LENGTH is set when the
        function is defined
        """

        # 
        buffer_count = buffer_length - len(local_string)

        if buffer_count < 0:
            return local_string

        return local_string + " "*buffer_count
    # ............................................................



    if silent:
        return

    print('Settings:')
    print("...................................")
    for name in kwargs:
        name_buffered = strbuffer(name)
        print(f"{name_buffered}: {kwargs[name]}")



# .................................................................
#
#
def validate_args(passed, valid, error_message=''):
    """
    General function that validates a specific string against a list of strings
    and raises an exception if the string ($passed) is not found in the list (valid).
    Good for validating passed options.

    Parameters
    -----------
    passed : str
        String of interest.

    valid : list
        List of strings

    Returns
    ----------
    None

    Raises
    -----------
    Exception
        If passed is not in valid, raise an exception
    
    """
    if passed not in valid:
        raise Exception(f'ERROR: Invalid option passed {passed}.\n{error_message}')


        
