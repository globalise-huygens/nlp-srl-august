import glob
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import ttest_ind
import json

#Small data scenario averages:
def averages_SFT(path_metrics):
    """
    Computes and prints average precision, recall, and F1 scores from a JSON file
    for single-task evaluation.

    Parameters
    ----------
    path_metrics : str
        Path to the JSON file containing single-task evaluation metrics.

    Returns
    -------
    tuple of lists
        (precision, recall, f1)
    """
    with open(path_metrics) as f:
        metrics = json.load(f)

    for dct in metrics: 
        precision = dct['metrics']['precision']
        recall = dct['metrics']['recall']
        f1 = dct['metrics']['f1']
        average_precision = sum(precision)/len(precision)
        average_recall = sum(recall)/len(recall)
        average_f1 = sum(f1)/len(f1)
    
    print(f'Average precision = {average_precision:.2f}')
    print(f'Average recall = {average_recall:.2f}')
    print(f'Average F1 = {average_f1:.2f}')

    return precision, recall, f1

def averages_MFT(path_metrics_mt):
    """
    Computes and prints average precision, recall, and F1 scores for both SRL and NER
    from a multitask metrics JSON file.

    Parameters
    ----------
    path_metrics_mt : str
        Path to the JSON file containing multitask evaluation metrics.

    Returns
    -------
    tuple of lists
        (srl_prec, srl_rec, srl_f1, ner_prec, ner_rec, ner_f1)
    """
    srl_prec = []
    srl_rec = []
    srl_f1 = []
    ner_prec = []
    ner_rec = []
    ner_f1 = []

    with open(path_metrics_mt) as f:
        metrics = json.load(f)

    for dct in metrics: 
        precision = dct['metrics']['precision']
        for p in precision:
            srl_prec.append(p['srl'])
            ner_prec.append(p['ner'])
        recall = dct['metrics']['recall']
        for r in recall:
            srl_rec.append(r['srl'])
            ner_rec.append(r['ner'])
        f1 = dct['metrics']['f1']
        for f in f1:
            srl_f1.append(f['srl'])
            ner_f1.append(f['ner'])
        average_precision = sum(srl_prec)/len(srl_prec)
        average_recall = sum(srl_rec)/len(srl_rec)
        average_f1 = sum(srl_f1)/len(srl_f1)
        avg_ner_prec = sum(ner_prec)/len(ner_prec)
        avg_ner_rec = sum(ner_rec)/len(ner_rec)
        avg_ner_f1 = sum(ner_f1)/len(ner_f1)
    print('SRL scores')
    print(f'Average precision = {average_precision:.2f}')
    print(f'Average recall = {average_recall:.2f}')
    print(f'Average F1 = {average_f1:.2f}')
    print()
    print('NER scores')
    print(f'Average precision = {avg_ner_prec:.2f}')
    print(f'Average recall = {avg_ner_rec:.2f}')
    print(f'Average F1 = {avg_ner_f1:.2f}')

    return srl_prec, srl_rec, srl_f1, ner_prec, ner_rec, ner_f1



#More Data scenario averages
def get_files(directory):
    """
    Recursively collects all .conllu files from a given directory and its subdirectories.

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



def compute_filtered_macro_scores(y_true, y_pred):
    """
    Computes macro-averaged precision, recall, and F1, ignoring labels with support=0.
    Returns: macro scores, accuracy, and per-class metrics for included labels.
    """
    labels = sorted(list(set(y_true + y_pred)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    mask = np.array(support) > 0
    filtered_labels = np.array(labels)[mask]
    filtered_precision = precision[mask]
    filtered_recall = recall[mask]
    filtered_f1 = f1[mask]
    filtered_support = np.array(support)[mask]

    macro_precision = np.mean(filtered_precision)
    macro_recall = np.mean(filtered_recall)
    macro_f1 = np.mean(filtered_f1)
    accuracy = accuracy_score(y_true, y_pred)

    per_class = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s)
        }
        for label, p, r, f, s in zip(filtered_labels, filtered_precision, filtered_recall, filtered_f1, filtered_support)
    }

    return macro_precision, macro_recall, macro_f1, accuracy, per_class

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


def calculate_averages(precision, recall, f1):
    """
    Compute and print averages of precision, recall, and F1.

    Parameters
    ----------
    precision : list of float
    recall : list of float
    f1 : list of float

    Returns
    -------
    None
    """
    average_precision = sum(precision)/len(precision)
    average_recall = sum(recall)/len(recall)
    average_f1 = sum(f1)/len(f1)

    print(f'precision: {average_precision:.2f}')
    print(f'recall: {average_recall:.2f}')
    print(f'f1: {average_f1:.2f}')


#calculate significance MD vs MD or SD vs SD (same length)
def calculate_significance_paired(scores_1, scores_2):
    """
    Performs a paired t-test to assess the statistical significance between
    two sets of model scores.

    Parameters
    ----------
    scores_1 : list of float
        Scores from the first model.
    scores_2 : list of float
        Scores from the second model.

    Returns
    -------
    None
    """
    a = np.array(scores_1)
    b = np.array(scores_2)

    # Mean and std
    mean_a, std_a = np.mean(a), np.std(a)
    mean_b, std_b = np.mean(b), np.std(b)

    print(f"Model SRL F1: {mean_a:.4f} ± {std_a:.4f}")
    print(f"Model Multi SRL F1: {mean_b:.4f} ± {std_b:.4f}")

    # Paired t-test
    t_stat, p_ttest = ttest_rel(a, b)
    print(f"Paired t-test p-value: {p_ttest:.4f}")

#calculate signficance MD vs SD (different length)
def significance_Welch(scores_1, scores_2):
    """
    Performs Welch’s t-test for independent samples of unequal sizes.

    Parameters
    ----------
    scores_1 : list of float
        Scores from the first model.
    scores_2 : list of float
        Scores from the second model.

    Returns
    -------
    None

    Notes
    -----
    Prints p-value of Welch’s independent t-test.
    """
    a = np.array(scores_1)
    b = np.array(scores_2)

    t_stat, p_ttest = ttest_ind(a, b, equal_var=False)  # Welch’s t-test
    print(f"Independent t-test p-value: {p_ttest:.4f}")

#get per class scores across files

def load_labels_from_file_task(path, task=None):
    """
    Used in compute_average_across_folds.
    Load Gold and Prediction columns from a file.
    If a task is specified, only keep rows where Task == task.

    Returns:
        y_true (list of str)
        y_pred (list of str)
    """
    df = pd.read_csv(path, sep="\t", comment="=", engine="python", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    if "Gold" not in df.columns or "Prediction" not in df.columns:
        raise ValueError(f"File {path} does not contain required columns 'Gold' and 'Prediction'.")

    # Filter by task if needed
    if task and "Task" in df.columns:
        df = df[df["Task"] == task]

    y_true = df["Gold"].dropna().astype(str).tolist()
    y_pred = df["Prediction"].dropna().astype(str).tolist()

    # Ensure same length
    n = min(len(y_true), len(y_pred))
    return y_true[:n], y_pred[:n]

def compute_average_across_folds(directory, pattern="complete_pred_versus_gold_fold*.txt",
                                 exclude_O=False, task=None, save_csv=True):
    """
    Uses the functions load_labels_from_file_task and compute_filtered_macro_scores.
    For each matching file in directory:
      - compute per-class metrics via compute_filtered_macro_scores
      - optionally only for a specific task
      - record per-class metrics for that fold
    Then average per-class metrics across folds (only folds where class had support).
    Returns avg_macro, avg_per_class_df, per_fold_details with final values rounded to 2 decimals
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern} in {directory}")

    per_fold_macro = []
    per_class_accumulator = {}  # label -> list of dicts from folds (precision/recall/f1/support)
    per_fold_details = []

    for fpath in files:
        y_true, y_pred = load_labels_from_file_task(fpath, task=task)

        if exclude_O:
            keep_mask = [(g != "O") or (p != "O") for g, p in zip(y_true, y_pred)]
            y_true = [g for g, k in zip(y_true, keep_mask) if k]
            y_pred = [p for p, k in zip(y_pred, keep_mask) if k]

        mp, mr, mf1, acc, per_class = compute_filtered_macro_scores(y_true, y_pred)
        per_fold_macro.append({"file": os.path.basename(fpath), "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1, "accuracy": acc})
        per_fold_details.append({"file": os.path.basename(fpath), "per_class": per_class})

        # accumulate per-class metrics
        for label, metrics in per_class.items():
            if exclude_O and label == "O":
                continue
            if label not in per_class_accumulator:
                per_class_accumulator[label] = {"precision": [], "recall": [], "f1": [], "support": []}
            per_class_accumulator[label]["precision"].append(metrics["precision"])
            per_class_accumulator[label]["recall"].append(metrics["recall"])
            per_class_accumulator[label]["f1"].append(metrics["f1"])
            per_class_accumulator[label]["support"].append(metrics["support"])

    # average per-class across folds (full precision)
    avg_per_class = {}
    for label, lists in per_class_accumulator.items():
        avg_per_class[label] = {
            "precision": np.mean(lists["precision"]) if lists["precision"] else 0.0,
            "recall":    np.mean(lists["recall"])    if lists["recall"] else 0.0,
            "f1":        np.mean(lists["f1"])        if lists["f1"] else 0.0,
            "support":   np.mean(lists["support"])   if lists["support"] else 0.0,
            "fold_count": len(lists["support"])
        }

    # average macro metrics across folds (full precision)
    avg_macro = {
        "macro_precision": np.mean([m["macro_precision"] for m in per_fold_macro]),
        "macro_recall":    np.mean([m["macro_recall"]    for m in per_fold_macro]),
        "macro_f1":        np.mean([m["macro_f1"]        for m in per_fold_macro]),
        "accuracy":        np.mean([m["accuracy"]        for m in per_fold_macro])
    }

    avg_macro = {k: round(v, 2) for k, v in avg_macro.items()}
    for label in avg_per_class:
        for metric in ["precision", "recall", "f1", "support"]:
            avg_per_class[label][metric] = round(avg_per_class[label][metric], 2)

    # convert avg_per_class to dataframe for saving/printing
    avg_per_class_df = (pd.DataFrame.from_dict(avg_per_class, orient="index")
                        .rename_axis("label")
                        .reset_index()
                        .sort_values(["f1", "label"], ascending=[False, True]))

    if save_csv:
        out_dir = directory
        avg_per_class_df.to_csv(os.path.join(out_dir, "avg_per_class_across_folds.csv"), index=False)
        pd.DataFrame(per_fold_macro).to_csv(os.path.join(out_dir, "per_fold_macro.csv"), index=False)
        for det in per_fold_details:
            perclass = det["per_class"]
            df = pd.DataFrame.from_dict(perclass, orient="index").rename_axis("label").reset_index()
            fname = f"per_class_{det['file'].replace('.','_')}.csv"
            df.to_csv(os.path.join(out_dir, fname), index=False)


    return avg_macro, avg_per_class_df, per_fold_macro
