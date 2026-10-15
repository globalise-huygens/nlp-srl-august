import glob
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import ttest_ind
import json


def get_files(directory):
    """
    Recursively collects all complete_pred_vs_gold{fold}.txt files from a given directory and its subdirectories.

    Parameters:
    directory (str): The root directory to search for .conllu files.

    Returns:
    list of str: A list of full file paths to all found .conllu files.
    """
    file_paths = []
    for path in glob.glob(os.path.join(directory, '**', 'complete*.txt'), recursive = True):
        if os.path.isfile(path):
            file_paths.append(path)
    return file_paths

def calculate_scores_per_fold(file_paths, task=None):
    """
    Compute macro precision, recall, and F1 for a list of prediction files (folds).

    Parameters
    ----------
    file_paths : list of str
        Paths to prediction files.
    task : str, optional
        If provided, only include rows matching this task.

    Returns
    -------
    tuple of lists
        precision : list of float, macro precision per fold
        recall : list of float, macro recall per fold
        f1_score : list of float, macro F1 per fold
    """
    precision = []
    recall = []
    f1_score = []

    for file in file_paths:
        df = pd.read_csv(file, sep="\t", comment="=", engine="python", skip_blank_lines=True, on_bad_lines="skip")
        df = df.dropna(subset=["Gold", "Prediction"])

        if task and "Task" in df.columns:
            df = df[df["Task"] == task]

        out = df[["Gold", "Prediction"]]
        gold_list = out["Gold"].tolist()
        pred_list = out["Prediction"].tolist()

        prec, rec, f1, acc, p_class = compute_filtered_macro_scores(gold_list, pred_list)
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)

    return precision, recall, f1_score


def calculate_mean_std(precision, recall, f1, ddof=1):
    """
    Compute and print mean ± std of precision, recall, and F1 across runs.

    Parameters
    ----------
    precision, recall, f1 : list of float
        One value per seed.
    ddof : int
        1 = sample std (default), 0 = population std.

    Returns
    -------
    dict : {metric: (mean, std)}
    """
    results = {}
    for name, values in [('precision', precision), ('recall', recall), ('f1', f1)]:
        values = np.asarray(values, dtype=float)
        mean, std = values.mean(), values.std(ddof=ddof)
        results[name] = (mean, std)
        print(f'{name}: {mean:.2f} ± {std:.2f}')
    return results


