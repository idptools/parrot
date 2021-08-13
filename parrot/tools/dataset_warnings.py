"""
Module with functions for checking if dataset meets common heuristics. Writes
warnings to console.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import numpy as np
from scipy.stats import skew

# Duplicate sequences:
def check_duplicate_sequences(d):
    seen = set()
    nonuniq = set()
    for x in d:
        if x[1] not in seen:
            seen.add(x[1])
        else:
            nonuniq.add(x[1])

    nonuniq = list(nonuniq)
    if len(nonuniq) > 0:
        print("\n#############################################")
        print("WARNING: Duplicate sequences detected in dataset.\n")
        print("Duplicated sequences:")
        for seq in nonuniq:
            print(seq)
        print()
        print("This may cause overfitting or overestimation of performance.\n")
        print("#############################################\n")

# Imbalanced classification dataset:
def check_class_imbalance(d):
    all_labels = np.hstack([x[2] for x in d])
    label_counts = {}
    for label in all_labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # Check for class imbalance:
    # Balanced = 1:1 - 1:1.4
    # Slight imbalance = 1:1.4 - 1:2.5
    # Severe imbalance = 1:2.5+
    # (These divisions are arbitrary)
    min_count = min(list(label_counts.values()))
    max_count = max(list(label_counts.values()))
    total_counts = np.sum(list(label_counts.values()))
    ratio = max_count / min_count

    # Slight imbalance
    if 1.4 < ratio <= 2.5:
        print("\n#############################################")
        print("WARNING: Slight class imbalance detected in dataset.\n")
        print("Class frequencies:")
        for label, count in label_counts.items():
            print(f'  {int(label)} : {np.around(100*count/total_counts, 1)}%')
        print()
        print("This may skew predictions towards overrepresented classes.\n")
        print("#############################################\n")

    # Severe imbalance
    elif 2.5 < ratio:
        print("\n#############################################")
        print("WARNING: Severe class imbalance detected in dataset.\n")
        print("Class frequencies:")
        for label, count in label_counts.items():
            print(f'  {int(label)} : {np.around(100*count/total_counts, 1)}%')
        print()
        print("Predictions will be strongly skewed towards overrepresented classes.")
        print("Classification not recommended with current dataset.\n")
        print("#############################################\n")

# Distribution of regression values
def check_regression_imbalance(d):
    vals = np.hstack([x[2] for x in d])
    check_zero_centered(vals)
    check_low_std(vals)
    check_skew(vals)

# Regression mean location
def check_zero_centered(v):
    # Using [-5, 5] is arbitrary, but seems to do the trick
    if np.mean(v) <= -5 or np.mean(v) >= 5:
        print("\n#############################################")
        print("WARNING: Dataset not centered around zero.\n")
        print(f"Data mean: {np.around(v.mean(), 2)}")
        print()
        print("This may decrease training efficiency.\n")
        print("#############################################\n")

# Regression standard deviation magnitude
def check_low_std(v):
    # Using >5 is arbitrary, but seems to do the trick
    if np.std(v) > 5:
        print("\n#############################################")
        print("WARNING: Dataset has large variance\n")
        print(f"Data standard deviation: {np.around(np.std(v), 2)}")
        print()
        print("This may decrease training efficiency.")
        print("Consider standardizing your dataset.\n")
        print("#############################################\n")       

# Regression distribution of values
def check_skew(v):
    s = skew(v)
    if s > 1:
        print("\n#############################################")
        print("WARNING: Dataset is skewed.\n")
        print()
        print("This may cause model bias.\n")
        print("#############################################\n")

# Batch size vs dataset size:
def eval_batch_size(batch, n_samples):
    if batch > 512:
        print("\n#############################################")
        print("WARNING: Using a very large batch size.\n")
        print()
        print("This may decrease model accuracy.\n")
        print("#############################################\n")

    if n_samples // batch <= 10:
        print("\n#############################################")
        print("WARNING: Batch size is large relative to the number of samples in the training set.\n")
        print()
        print("This may decrease training efficiency.\n")
        print("#############################################\n")
