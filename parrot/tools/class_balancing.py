"""
Module with functions for dealing with automatic class balacing

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import random
import numpy as np
import datetime


from parrot.parrot_exceptions import ParrotException 

EMERGENCY_COUNT_THRESHOLD=10



def __validate_class_balance_input(balance_threshold, max_fraction_removed):
    """
    Helper function that checks that a balance_threshold and a max_fraction_removed threshold

    Parameters
    -------------
    balance_threshold : float
        Float that reports on the balance score threshold to be used. Expects to be between 1 and 5.0

    max_fraction_removed : float
        Fraction which defines the max fraction of the data that can be removed in the name of 
        class balancing. Expects to be between 0 and 1 not inclusive.


    Returns
    ----------
    None
        No return

    Raises
    ----------
    parrot.parrot_exceptions.ParrotException
        If an invalid exception then raises a ParrotException

    """
    
    if balance_threshold < 1.0 or balance_threshold > 5:
        raise ParrotException('Error during class balancing. The balance threshold must be between 1.0 and 5 - asking to balance with a balance score > 5 does not any sense')

    if max_fraction_removed <= 0 or max_fraction_removed >= 1.0:
        raise ParrotException('Error during class balancing. max_fraction_removed must be between 0 and 1 (not inclusive)')
    

# ..........................................................................................
#
#
def parse_class_info(lines):
    """
    Function that extracts out class information from a set of parsed datafile
    lines. Lines is a list of lists and should be generated by the function
    parrot.process_input_data.__read_tsv(<filename>).

    The function takes in a files list and returns a tuple with two bits of 
    information; the list of classes found in the file, and the mapping between
    sequence index and entry. Note sequence index is a numerical internal index
    that starts at 0 for the first line and increments accordingly, and does NOT
    correspond to the identifier used in the data file.

    Parameters
    ------------
    lines : list
        Lines is a specific type of list, defined as the output from the function
        parrot.process_input_data.__read_tsv(<filename>).

    Returns
    ----------
    tuple
        The function returns a tuple with the following two elements
    
        [0] - list of the classes (where classes are strings as defined in the 
              lines list)

        [1] - a dictionary that maps index (starting at 0) to a second dictionary,
              which itself maps the class name to the number of instances that
              class pops up. This dictionary is the idx2classes dict which is
              referenced through this file
    
    """
    original_data = {}
    classes = set([])
    # first figure out the number of classes
    idx=0
    for line in lines:
        classes = classes.union(set(line[2:]))
        original_data[idx] = line
        idx = idx + 1

    # order classes in a consistent manner
    n_classes = len(classes)
    classes = list(classes)
    classes.sort()
    
    idx2classes = {}
    for idx in original_data:
        tmp = {}
        for c in classes:
            tmp[c] = original_data[idx][2:].count(c)
        idx2classes[idx] = tmp

    return (classes,idx2classes)


# ..........................................................................................
#
#
def get_total_class_count(classes, idx2classes, old_total_class_count=None, removed_entry_dict=None):
    """
    Function which calculates the counts for each of the classes - i.e. given the idx2classes dictionary, how many times
    does each of the classes (defined by the classes list) appear?
    
    
    This can be calculate in one of two ways:

    (1) Directly from the idx2classes dictionary with the classes list. This is computationally a little expensive
        in that each entry in idx2classes dictionary, but gives a ground truthe.

    (2) Using an old total_class_count dictionary and a removed_entry_dict which reflects a single entry that is to
        be removed. To expand a bit on this, if an old_total_class_count dictionary is provided, it should be a dictionary
        of the format {'class name':class_count,...}. Then, a removed_entry_dict should be of the same format, an the
        returned total_class_count dictionary subtracts the counts for the removed_entry_dict from the old_total_class_count
        dictionary. 

    The first option is a de novo calculation, whereas the second one is an almost instantaneous correction, although we 
    do allocate a new dictionary so the return variable is a different instantiation than the old_total_class_count.
 
    Parameters
    -------------
    classes : list
        List of valid classes. All classes in the idx2classes dictionary should be represented in the 
        classes list, and, similarly, every dictionary in the idx2classes should map between each of 
        the classes and a count (even if the count is 0).
        
    idx2classes : dict
        Dictionary that maps index (starting at 0) to a second dictionary, which itself maps the 
        class name to the number of instances that class pops up.

    old_total_class_count : dict
        If provided, this should be a single dictionary that maps class to total count. 
        Default = None.

    removed_entry_dict : dict
        If provided, this should be a single dictionary that maps class to a count. The counts here
        are subtracted from the old_total_class_count


    Returns
    ------------
    dict
        Dict which maps between class identity and total counts for that class

    Raises
    ------------
    Exception
    
            
    """

    # if both removed entry dict and old_total_class_count are provdied
    if removed_entry_dict is not None and old_total_class_count is not None:

        # update the old_total_class_count and return - note we do allocate a 
        # new dictionary in case this function ever is used in a scenario where
        # pass by reference would be a problem        
        new_dict = {}
        for c in old_total_class_count:
            new_dict[c] = old_total_class_count[c] - removed_entry_dict[c]
            
        return new_dict
    

    # if only one of these two is allocate through an exception
    if removed_entry_dict is not None or old_total_class_count is not None:
        raise Exception('If removed_entry_dict is provided then old_total_class_count must also be provided (and vice versa)')
    

    # If we get here we're calculating the total_class_count from scratch
    total_class_count = {}

    # initialize
    for c in classes:
        total_class_count[c] = 0
        
    # for each entry
    for idx in idx2classes:
        for c in classes:
            total_class_count[c] = total_class_count[c] + idx2classes[idx][c]
                        
    return total_class_count


# ..........................................................................................
#
#
def calculate_imbalance(total_class_count, idx_to_remove=None, idx2classes=None):
    """
    Function that calculates a simple metric for class imbalance, which is just the ratio
    of the most abundant class over the least abundant class.

    A perfectly balanced dataset would have an imbalance score of 1.0. 

    If idx_to_remove and idx2classes are provided, the imbalance score is calculated
    AFTER the idx_to_remove was removed from the total_class_count
    
    Parameters
    ----------------
    total_class_count : dict
        Dict which is total count for each class. This is the output from 
        get_total_class_count.
        
    idx_to_remove : int
        If provided, defines the index of the entry that is to be removed
        prior to imbalance calculation. Default = None.
        
    idx2classes : dict
        Dict that maps an index to a second dictionary which says for that
        sequence what the count of each class is. Default = None.

    Returns
    -------------
    float
        Returns a float which is the ratio of the largest to smallest class. If idx_to_remove
        is passed then calculates what this would be after that idx was removed (but does not)
        actually remove it!

    
    """
        
    # if we're asking how removing a specific index would change the imbalance...
    if idx_to_remove is not None and idx2classes is not None:
        local_class_counts = idx2classes[idx_to_remove]
        new_total_class_count = {}
        
        for c in local_class_counts:
            new_total_class_count[c] = total_class_count[c] - local_class_counts[c]

    elif idx_to_remove is None and idx2classes is  None:
        new_total_class_count = total_class_count

    else:
        raise Exception('If idx_to_remove is passed must also pass idx2classes')


        
    biggest_idx = max(new_total_class_count, key=new_total_class_count.get)
    smallest_idx = min(new_total_class_count, key=new_total_class_count.get)

    return (new_total_class_count[biggest_idx] / new_total_class_count[smallest_idx])


        
# ..........................................................................................
#
#    
def run_class_balance_residues(lines, 
                               balance_threshold=1.4,                                
                               block_size=100, 
                               max_fraction_removed=0.25, 
                               shuffle_data=True,
                               verbose=True):

    """
    Function that takes in a read lines dataset (from parrot-preprocess) and removes
    a subset of the elements such that the dataset is class balanced. This implements the 
    Stochastic Residue Class Balancing (SRCB) algorithm, an approach to classified sequence
    data to generate a more balanced dataset.

    Specifically, SRCBworks by doing the following:

    1. Calculating an inbalance score (ratio of most abundant class over least abundant class)

    2. Asking if this value is below the balance_threshold - if yes, exit and we're done
    
    3. If no, we cycle through the data for a fixed number of iterations. Each iteration we 
       randomly select a set of block_size datapoints, and for each of those datapoints ask 
       how the imbalance score would be changed if each of them were removed. We find the datapoint
       that leads to the greates decrease in the imbalance score and then remove that data point.

    4. This process is repeated unill EITHER we removed the max number of datapoints we allow, as
       defined by max_fraction_removed, OR our new dataset has a sufficiently low imbalance score. 
       
    The big advantage of SRCB  is that it scales extremely well with data size because each
    iteration in the removal step ONLY compares a fixed (block_size) number of datapoints, 
    and changes to the imbalance score can then be calculated based on a change in the previous
    values, such that we basically avoid iteracting over the entire dataset entirely other than
    during an intial shuffle operation to remove correlations in the data and when constructing
    the initial counts for classes. This approach runs in under ~30 seconds for datasets of 
    100s of thousands of datapoints. As block_size gets bigger and max_fraction_removed gets
    larger than algoirthm will slow approximately linearly with both parameters.

    In addition to the parameters passed into this function, there's a hidden parameter defined
    as a static const in this file called EMERGENCY_COUNT_THRESHOLD. This parameter defines how 
    many times we try and find a block that improves the imbalance score - for example. If after
    EMERGENCY_COUNT_THRESHOLD attempts we stil don't find a datapoint that improves the current
    balance we'll remove a datapoint that makes it the 'least bad'. This ensures the algorithm
    cannot get stuck in an infite loop. Right now this value is set to 10, but, if it should
    be set to something more permissive I'm not sure.

    Parameters
    -------------
    lines : list
        Lines is a list of lists, where each sublist is of the format [<id>, <sequence>, <class1>, ...,<classn>]. The 
        number of class1,class2 data points should match the number of residues in the <sequence>.

    balance_threshold : float
        A value that reports on how much imbalance is tolerated. The imbalance score is just the ratio of the 
        count for the most frequenct class divided by the count of the least frequent class. i.e. a score of 1.0 would
        mean the dataset is perfectly balanced. We use a threshold of 1.4 to match the warning thresholds defined in
        parrot.tools.dataset_warnings, but these could be updated. Default = 1.4

    block_size : int
        Parameter that defines the size of the local block size. In practice, if block_size were set to the data
        size this algorithm would GUARENTEE the optimally balanced dataset. In practice, using a block size of ~0.1%
        of the data (e.g. 100 data points in a dataset of 100,000) has been reasonable in achieving an almost
        identical d(imbalance score)/d(number of data points removed) for 'real' data. Default 100.

    max_fraction_removed : float
        What fraction of the data can be removed in the persuit of a balance dataset? Must be between 0 and 1 (not
        inclusive of either). Default = 0.25.

    shuffle_data : bool
        Flag which, if set to true, means data is fully shuffled before sub-sampling is done. This is generally
        good in that the SRCB algorithm assumes no correlation in data, but there may be times in which retaining
        the approximate structure of the input data (with a subset of lines removed) is preferable, so, this
        gives some control over that.

    verbose : bool
        Flag which indicates if status and input info should be printed to screen.

    Returns
    -----------
    (list, list)
        Returns a tuple of two lists:

        [0] - The first list is the down-sampled data in the format of a standard lines list (i.e. matches the
              lines input variable).

        [1] - Trace of balance score over each iteration. Only includes iterations where a datapoint
              was removed. Useful for assessing how changing the parameters influenced the trajectory
              if data exclusion. 

    """

    # validate input...
    if block_size < 2:
        raise ParrotException('Error during class balancing. Block size must be greater than 1')

    if block_size > 0.2*len(lines):
        print(f'Warning: Block size is unecessarily large - reducing to {int(0.2*len(lines))} [10% of data size].')
        block_size = int(0.2*len(lines))

    __validate_class_balance_input(balance_threshold, max_fraction_removed)

    # shuffle lines - we do this so any intrinsic structure in the input data. The ClassBalance alrg
    if shuffle_data:
        random.shuffle(lines)

    # parse input 
    (classes, idx2classes) = parse_class_info(lines)

    # initial class imbalance score
    total_class_count = get_total_class_count(classes, idx2classes)
    original_imbalance_score = calculate_imbalance(total_class_count=total_class_count)

    itercount = int(len(idx2classes)*max_fraction_removed)

    # cross-check itercount to make sure it's a useable number
    if itercount < 1:
        itercount = 1

    if itercount > len(idx2classes):
        itercount = len(idx2classes) - 1
    

    if verbose:
        print('-------------------------------------------------------------------')
        print('Performing automatic class balancing for residues (--class-balance)') 
        print('-------------------------------------------------------------------')
        print('')
        print(f'   Found {len(classes)} different classes accross {len(idx2classes)} sequences')
        print('   Class breakdown:')
        print("   " + str(total_class_count))
        print(f'   These classes have an imbalance ratio of {original_imbalance_score}')
        print(f'   Preparing to perform class balancing with the following parameters')
        print(f'   Max num. of removable sequences : {itercount}')
        print(f'   Imbalance threshold             : {balance_threshold}')
        print(f'   Block size                      : {block_size}')
        print(f'   Starting at {datetime.datetime.now()}') 


    # initialize a bunch of stuff
    removed_entry = None
    tracker = []
    tracker.append(original_imbalance_score)
    c = 0 
    emergency_count = 0
    recalculate_total_class_count = True

    # main whileloop of SRCB
    while c < itercount:

        # the emergency count incremenets on each iteraction and helps us 
        # avoid getting stuck in a while loop.
        emergency_count = emergency_count + 1

        # get the current set of possible indices we have to chose from in the
        # dataset. Recall indices here correspond to line numbers in the input 
        # lines list.
        valid_keys = list(idx2classes.keys())

        # select a random value between 0 and the number of keys minus the block size
        random_start = random.randint(0, len(valid_keys) - (1+block_size))

        # select a random value between the random start and the end of the number of keys
        random_end = random_start + block_size

        # finally, select a selection of indices based on the random start/end 
        # positions. 
        subset = valid_keys[random_start:random_end]

        # calculate total class count and total number of residues. If this is a perturbation from a previously
        # calculated value we use the prior state rather than recomputing everything. Note we only do this if
        # recalculate_total_class_count is set to True. If nothing was removed we don't need to recalculate
        # this as it hasn't changed.
        if recalculate_total_class_count:
            if removed_entry is not None:
                total_class_count = get_total_class_count(classes, idx2classes, old_total_class_count=total_class_count, removed_entry_dict=removed_entry)
            else:
                total_class_count = get_total_class_count(classes, idx2classes)

        # if we have reached the threshold...
        if tracker[-1] < balance_threshold:
            break

        # calculate how imbalance score would change if we removed each of the datapoints defined by the list of indices
        # in the subset list
        hits = {}
        for idx in subset:
            inbalance_score = calculate_imbalance(total_class_count=total_class_count, idx_to_remove=idx, idx2classes=idx2classes)
            hits[idx] = inbalance_score

        # get the index of the datapoint from subset that leads to the imabalance score to be the smallest
        offending = min(hits, key=hits.get)

        # only remove this entry if it makes things better or we've been stuck on the same 
        
        if hits[offending] < tracker[-1] or emergency_count % EMERGENCY_COUNT_THRESHOLD == 0:
            
            if emergency_count % EMERGENCY_COUNT_THRESHOLD == 0:                
                if verbose:
                    print(f'WARNING: Got stuck trying to improve the balance score on iterations {c} [tried {emergency_count} times).\nConsider re-running with a lower fraction to remove or a larger block-size')

            # remove data point
            removed_entry = idx2classes.pop(offending)    

            # increment counter
            c = c + 1

            # reset emergency count counter and recalculate_total_class_count flag
            emergency_count = 0
            recalculate_total_class_count = True

            # update the tracker
            tracker.append(calculate_imbalance(total_class_count=total_class_count))
        else:
            recalculate_total_class_count = False



    # note this works because the indices in idx2classes correspond to the line
    # number in the original lines input list
    new_lines = []
    for idx in list(idx2classes.keys()):
        new_lines.append(lines[idx])
        
    if verbose:
        frac_removed = c/len(lines)
        print('   -------------------------------------')
        print(f'   Complete at {datetime.datetime.now()}') 
        print(f'   Removed {c} sequences ({frac_removed:.3f} of sequences)')
        print(f'   Updated balance score: {tracker[-1]}')
        print(f'   Updated class breakdown:')
        print("   "+str(total_class_count))
        print('   -------------------------------------\n')
        


    return (new_lines, tracker)
    

        

# ..........................................................................................
#
#    
# Note - most of this module is concerned wth run_class_balance_residue - the sequence-based
# class balancing is much easier and is dealt with internally here
#
#
def run_class_balance_sequences(lines, 
                                balance_threshold=1.4, 
                                max_fraction_removed=0.25, 
                                verbose=True):

    """
    Function that performs class balancing if classes are set at the sequence level. This uses a completely
    different algorithm to the SRCB (for sequences) as this is a much simpler problem. This algorithm here
    uses the following approach:


    1. Groups sequences based on their class
    2. Calculates imbalance score (count for most abundant class / count for least abundant class)
    3. Calculates the count for the most abundant class and the average count for the remaining classes
    4. Subtracts random datapoints from the most abundant class to match the average of the remainders
    5. Recalculate and check how many data points were removed
    6. Repeat or exit

    This is a much more simple and much faster algorithm the the SRCB. It still scales well with arbitrarily
    large datasets, in that in general only a few iterations are required for convergence because each iteration
    removes many datapoints.

    Parameters
    -------------
    lines : list
        Lines is a list of lists, where each sublist is of the format [<id>, <sequence>, <class>]. This in general
        should be generated by the function parrot.process_input_data.__read_tsv().

    balance_threshold : float
        A value that reports on how much imbalance is tolerated. The imbalance score is just the ratio of the 
        count for the most frequenct class divided by the count of the least frequent class. i.e. a score of 1.0 would
        mean the dataset is perfectly balanced. We use a threshold of 1.4 to match the warning thresholds defined in
        parrot.tools.dataset_warnings, but these could be updated. Default = 1.4
    

    max_fraction_removed : float
        What fraction of the data can be removed in the persuit of a balance dataset? Must be between 0 and 1 (not
        inclusive of either). Default = 0.25.

    verbose : bool
        Flag which indicates if status and input info should be printed to screen.
    
    Returns
    -----------
    list
        Down-sampled lines-list data in the format of a standard lines list (i.e. matches the
        lines input variable). 

    """

    
    # ================================================================================================
    def __build_total_class_count(c2l):
        """
        Internal function that converts a class2line dictonary to a total_class_count dictionary.
        This is defined as an internal function because its only appropriate to work with sequence
        classes, not residues classes.

        Parameters
        ------------
        c2l : dict
            The class2dict dictionary has class names as keys and values are lists of lines that
            correspond to that class.

        Returns
        ----------
        dict
            Returns a total_class_count dictionary, where keys are class names and values are number
            of times a line with that class appears.
        
        """
    
        new_total_class_count = {}
        for x in c2l:
            new_total_class_count[x] = len(c2l[x])
        
        new_total_class_count = {k: v for k, v in sorted(new_total_class_count.items(), key=lambda item: item[1], reverse=True)}
        return new_total_class_count
    # ================================================================================================

    
    # validate input options (same validation as used for run_class_balance_residues())
    __validate_class_balance_input(balance_threshold, max_fraction_removed)



    # shuffle lines
    random.shuffle(lines)
    

    # maps class name => list of lines that have datapoints for that class
    class2line = {}

    # maps class name => count 
    total_class_count = {} 
    
    # build the class2line dictionary and total_class_count dictionary
    for line in lines:
        c = line[2]

        if c not in class2line:
            class2line[c] = []
            total_class_count[c] = 0

        class2line[c].append(line)
        total_class_count[c] = total_class_count[c] + 1

    # this generates an ordered dictionary where the keys are largest-to-smallest ordered by count
    total_class_count = {k: v for k, v in sorted(total_class_count.items(), key=lambda item: item[1], reverse=True)}

    # calculate the max number of datapoints we're allowed to remove
    max_removal = int(len(lines)*max_fraction_removed)
        
    # calculate initial class balance score (count of biggest class/count of smallest class)
    class_balance_score = total_class_count[list(total_class_count.keys())[0]]/total_class_count[list(total_class_count.keys())[-1]]

    # set the all-time-removal counter
    all_time_removal = 0

    if verbose:
        print('-------------------------------------------------------------------')
        print('Performing automatic class balancing for sequences (--class-balance)') 
        print('-------------------------------------------------------------------')
        print('')
        print(f'   Found {len(total_class_count)} different classes accross {len(lines)} sequences')
        print('   Class breakdown:')
        print("   "+str(total_class_count))
        print(f'   These classes have an imbalance ratio of {class_balance_score}')
        print(f'   Preparing to perform class balancing with the following parameters')
        print(f'   Max num. of removable sequences: {max_removal}')
        print(f'   Imbalance threshold            : {balance_threshold}')
        print(f'   Starting at {datetime.datetime.now()}\n') 

    # main loop
    while class_balance_score > balance_threshold:

        # calculate current dataset size (this will change as we remove datapoints
        dataset_size = np.sum([total_class_count[x] for x in total_class_count])

        # determine the name of the largest and smallest classt 
        largest_class = list(total_class_count.keys())[0] # class name
        smallest_class = list(total_class_count.keys())[-1] # class name
    
        # determine the names of all the classes that are not the largest class
        not_largest_class = list(total_class_count.keys())[1:] 
    
        # compute average count of all classes except the largest class
        average_of_not_largest_class = np.mean([total_class_count[x] for x in not_largest_class])
    
        # diff is the difference between the largest class and the average of the other classes. Note we use the
        # min of 1 or this value because this avoids an edge case where the diff would round to 0 if the mean
        # was the same...
        diff = np.max([1,total_class_count[largest_class] - average_of_not_largest_class])
    
        # remove EITHER the diff between largest and average OR the maximum number of 
        # datapoints we can remove in this round, whichever is smallest
        local_removal_count = int(np.min([diff,max_removal-all_time_removal]))

        # remove a set of datapoints from the largest cluster. We shuffled everything to start
        # so this statistically is just randomly removing data, even though its the first $n$
        # from this dataset
        for i in range(local_removal_count):
            class2line[largest_class].pop()
    
        # update the total_class_count (recall key order is defined by class count and goes largest to smallest)
        total_class_count = __build_total_class_count(class2line)
    
        # update the count of how many data points we've removed
        all_time_removal = all_time_removal + local_removal_count

        # recalculate the class imbalance score
        class_balance_score = total_class_count[list(total_class_count.keys())[0]]/total_class_count[list(total_class_count.keys())[-1]]
    
        # this gives us a threshold where we exist when we removed as many datapoints as we're allowed
        if local_removal_count == (max_removal - all_time_removal):
            break


    # finally construct return data and SHUFFLE the data so we don't accidentally write out a structured file which
    # would be problematic down the line
    new_lines = []
    for c in class2line:
        new_lines.extend(class2line[c])

    random.shuffle(new_lines)

    if verbose:
        frac_removed = all_time_removal/len(lines)
        print('   -------------------------------------')
        print(f'   Complete at {datetime.datetime.now()}') 
        print(f'   Removed {all_time_removal} sequences ({frac_removed:.3f} of sequences)')
        print(f'   Updated balance score: {class_balance_score}')
        print(f'   Updated class breakdown:')
        print("   "+str(total_class_count))
        print('   -------------------------------------\n')

    return new_lines
        
