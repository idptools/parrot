import parrot

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
        print("...................................")
        print("|                                 |")
        print("|    _        Welcome to PARROT!  |")
        print("|  /` '\                          |")    
        print("|/| @   l                         |")
        print("|\|      \                        |")
        print("|  `\     `\_                     |")
        print("|    \    __ `\                   |")
        print("|    l  \   `\ `\__               |")
        print("|     \  `\./`     ``\            |")
        print("|       \ ____ / \   l            |")
        print("|         ||  ||  )  /            |")
        print("|-------(((-(((---l /-------------|")
        print("|                l /              |")
        print("|               / /               |")
        print("|              / /                |")
        print("|             //                  |")
        print("|            /                    |")
        print("...................................")

    print(f'\nparrot-predict (version {parrot.__version__})')
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
def write_datafile(filename, outdata):
    """
    Function that writes out a parrot datafile, where outdata must
    be a list of lists and each sub-element is joined on a whitespace

    Parameters
    --------------
    filename : str
        The name of the file to be written to

    outdata : list
        A list of lists (e.g. lines in parrot-preprocess)

    Returns
    ---------
    None
        No returns but the outdata is written to the current file

    """
    with open(filename,'w') as fh:
        for line in outdata:
            fh.write(" ".join(line)+"\n")


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


        
