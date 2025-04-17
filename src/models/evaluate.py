import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from config.data_config import (
    SELECTED_TAGS, GROUND_TRUTH_TAG_COLUMNS,
    PREDICTION_TAG_COLUMNS
)

def evaluate_predictions(y_true_all, y_pred_all):
    """
    Compute precision, recall, and F1 for each tag and macro average F1.
    Also reports:
      - number of problems with perfect tag prediction
      - number with at least one correct tag
      - number with no incorrect tags and at least one correct one
      - percentage versions of the above counts

    Args:
        y_true_all (np.ndarray): Numpy array of shape (n_samples, n_tags) with ground truth labels.
        y_pred_all (np.ndarray): Numpy array of shape (n_samples, n_tags) with predicted labels.

    Returns:
        dict: Dictionary containing per-tag metrics, macro F1, and custom stats.
    """
    per_tag_metrics = {}
    f1_scores = []

    if y_true_all.shape != y_pred_all.shape:
        raise ValueError(f"Shape mismatch between y_true ({y_true_all.shape}) and y_pred ({y_pred_all.shape})")
    if y_true_all.shape[1] != len(SELECTED_TAGS):
        raise ValueError(f"Number of columns in y_true ({y_true_all.shape[1]}) does not match SELECTED_TAGS ({len(SELECTED_TAGS)})")


    print("\n--- Per-Tag Metrics ---")
    for i, tag in enumerate(SELECTED_TAGS):
        y_true = y_true_all[:, i]
        y_pred = y_pred_all[:, i]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        support = int(np.sum(y_true)) # Get the number of true positive examples for this tag

        per_tag_metrics[tag] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support
        }
        f1_scores.append(f1)
        print(f"Tag: {tag:<15} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Support: {support}")

    # Calculate Macro F1 (average of F1 scores across all tags)
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    print(f"\nMacro Average F1 Score: {macro_f1:.3f}")

    # Calculate Micro Averaged Metrics (treating all predictions as one large set)
    micro_precision = precision_score(y_true_all.ravel(), y_pred_all.ravel(), zero_division=0)
    micro_recall = recall_score(y_true_all.ravel(), y_pred_all.ravel(), zero_division=0)
    micro_f1 = f1_score(y_true_all.ravel(), y_pred_all.ravel(), zero_division=0)
    print(f"Micro Average Precision: {micro_precision:.3f}")
    print(f"Micro Average Recall: {micro_recall:.3f}")
    print(f"Micro Average F1 Score: {micro_f1:.3f}")


    # Custom metrics (Instance-based)
    exact_match_count = 0
    at_least_one_correct = 0
    no_wrong_tags_and_one_correct = 0
    total_samples = len(y_true_all)

    if total_samples == 0:
        print("\nWarning: No samples found for custom evaluation stats.")
        extra_stats = {
            "total_samples": 0,
            "perfect_match_count": 0, "perfect_match_percent": 0.0,
            "at_least_one_correct": 0, "at_least_one_correct_percent": 0.0,
            "no_wrong_and_some_correct": 0, "no_wrong_and_some_correct_percent": 0.0
        }
    else:
        for y_true, y_pred in zip(y_true_all, y_pred_all):
            is_exact_match = np.array_equal(y_true, y_pred)
            has_correct_tag = np.any((y_true == 1) & (y_pred == 1))
            has_no_wrong_tags = np.all((y_pred == 0) | (y_true == 1)) # True if all predicted 1s are correct 1s

            if is_exact_match:
                exact_match_count += 1
            if has_correct_tag:
                at_least_one_correct += 1
            if has_no_wrong_tags and has_correct_tag:
                no_wrong_tags_and_one_correct += 1

        extra_stats = {
            "total_samples": total_samples,
            "perfect_match_count": exact_match_count,
            "perfect_match_percent": round(100.0 * exact_match_count / total_samples, 2),
            "at_least_one_correct": at_least_one_correct,
            "at_least_one_correct_percent": round(100.0 * at_least_one_correct / total_samples, 2),
            "no_wrong_and_some_correct": no_wrong_tags_and_one_correct,
            "no_wrong_and_some_correct_percent": round(100.0 * no_wrong_tags_and_one_correct / total_samples, 2),
        }

        print("\n--- Custom Evaluation Stats ---")
        print(f"Total problems evaluated: {extra_stats['total_samples']}")
        print(f"Perfectly matched problems: {extra_stats['perfect_match_count']} ({extra_stats['perfect_match_percent']}%)")
        print(f"Problems with at least one correct tag predicted: {extra_stats['at_least_one_correct']} ({extra_stats['at_least_one_correct_percent']}%)")
        print(f"Problems with no incorrect tags predicted AND at least one correct tag: {extra_stats['no_wrong_and_some_correct']} ({extra_stats['no_wrong_and_some_correct_percent']}%)")

    # --- Combine results into a dictionary ---
    evaluation_results = {
        "per_tag_metrics": per_tag_metrics,
        "macro_average_f1": macro_f1,
        "micro_average_metrics": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1_score": micro_f1,
        },
        "custom_stats": extra_stats
    }

    return evaluation_results


def run_evaluation(predictions_csv_path: str) -> dict:
    """
    Loads predictions and ground truth from a CSV and computes evaluation metrics.

    Args:
        predictions_csv_path (str): Path to the CSV file containing predictions.
                                     Must contain ground truth columns (from GROUND_TRUTH_TAG_COLUMNS)
                                     and prediction columns (from PREDICTION_TAG_COLUMNS).

    Returns:
        dict: Dictionary with evaluation results.
    """
    print(f"\n--- Running Evaluation on: {predictions_csv_path} ---")
    if not os.path.exists(predictions_csv_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv_path}")

    df = pd.read_csv(predictions_csv_path)

    # Verify necessary columns exist
    missing_gt = [col for col in GROUND_TRUTH_TAG_COLUMNS if col not in df.columns]
    missing_pred = [col for col in PREDICTION_TAG_COLUMNS if col not in df.columns]

    if missing_gt:
        raise ValueError(f"Missing required ground truth columns in prediction file: {missing_gt}")
    if missing_pred:
        raise ValueError(f"Missing required prediction columns in prediction file: {missing_pred}")

    # Extract numpy arrays
    y_true_all = df[GROUND_TRUTH_TAG_COLUMNS].values
    y_pred_all = df[PREDICTION_TAG_COLUMNS].values

    # Ensure they are integer type
    y_true_all = y_true_all.astype(int)
    y_pred_all = y_pred_all.astype(int)

    results = evaluate_predictions(y_true_all, y_pred_all)
    print("--- Evaluation Complete ---")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth.")
    parser.add_argument(
        "predictions_csv_path",
        type=str,
        help="Path to the CSV file containing ground truth and prediction columns."
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default=None,
        help="Optional path to save the evaluation results as a JSON file."
    )

    args = parser.parse_args()

    evaluation_metrics = run_evaluation(args.predictions_csv_path)

    if args.output_json_path:
        # Determine output directory and create if needed
        output_dir = os.path.dirname(args.output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save the results
        with open(args.output_json_path, "w") as f:
            json.dump(evaluation_metrics, f, indent=4)
        print(f"\nEvaluation results saved to: {args.output_json_path}")