import os
from pathlib import Path

from typing import List, Tuple


def get_ninja_split(root: str | Path):
    studies = list(sorted(os.listdir(str(root))))
    return studies, studies


def split_data_into_folds(root: str | Path, n_studies_in_val: int = 2) -> List[Tuple[List[str], List[str]]]:
    """
    Split the data into folds, ensuring that validation part for any fold gets n_studies_in_val studies.
    """
    studies = list(sorted(os.listdir(str(root))))
    if len(studies) < n_studies_in_val:
        raise ValueError(
            f"Number of studies ({len(studies)}) is less than the number of studies in the validation set ({n_studies_in_val})."
        )

    n_studies = len(studies)

    folds = []
    for i in range(n_studies):
        # The validation set is a slice of size n_studies_in_val
        val_indices = [i % n_studies for i in range(i, i + n_studies_in_val)]
        val_studies = [studies[j] for j in val_indices]

        # All other indices go into the training set
        train_studies = [studies[j] for j in range(n_studies) if j not in val_indices]

        folds.append((train_studies, val_studies))

    return folds


def split_data_into_folds_leave_one_out(root: str | Path) -> List[Tuple[List[str], List[str]]]:
    """
    Split the data into folds, using leave-one-out strategy.
    """
    studies = list(sorted(os.listdir(str(root))))

    n_studies = len(studies)
    n_folds = n_studies

    folds = []

    for i in range(n_folds):
        # The validation set is a slice of size n_studies_in_val
        val_indices = [i]
        val_studies = [studies[j] for j in val_indices]

        # All other indices go into the training set
        train_studies = [studies[j] for j in range(n_studies) if j not in val_indices]

        folds.append((train_studies, val_studies))

    return folds

def custom_tomogram_split(root: str | Path) -> List[Tuple[List[str], List[str]]]:
    """
    Custom split that trains on specific tomograms (TS_5_4, TS_69_2, TS_6_4, TS_6_6, TS_73_6, TS_86_3)
    and evaluates on TS_99_9 from czii-cryo-et-object-identification dataset.
    
    Args:
        root: Root directory containing tomogram data
        
    Returns:
        List of folds (there's only one fold in this case)
    """
    # Specific tomograms for training
    train_tomograms = ['TS_5_4', 'TS_69_2', 'TS_6_4', 'TS_6_6', 'TS_73_6', 'TS_86_3']
    
    # Validation tomogram
    val_tomograms = ['TS_99_9']
    
    # Return as a single fold - this is the only fold we'll use
    return [(train_tomograms, val_tomograms)]
