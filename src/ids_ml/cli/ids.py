"""
Intrusion Detection System (IDS) - Command Line Interface

This CLI tool uses a trained XGBoost model to detect harmful traffic in network flows.
"""

import argparse
import json
import sys

import pandas as pd

from ids_ml.models.inference import load_model, predict


def is_harmful(label):
    """Determine if a traffic label indicates harmful activity."""
    return label.upper() != "BENIGN"


def analyze_traffic_file(
    file_path, model, label_mapping, scaler=None, selector=None, verbose=False, output_file=None
):
    """
    Analyze a CSV file containing network traffic flows.

    Args:
        file_path: Path to CSV file with traffic data
        model: Loaded XGBoost model
        label_mapping: Dictionary mapping class indices to labels
        scaler: Optional MinMaxScaler for preprocessing
        selector: Optional SelectKBest selector for feature selection
        verbose: If True, print detailed information for each flow
        output_file: Optional path to save detailed results as CSV
    """
    print(f"\n{'=' * 70}")
    print(f"Analyzing traffic from: {file_path}")
    print(f"{'=' * 70}\n")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty - {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if "Label" in df.columns:
        print("Note: Removing existing 'Label' column from input data\n")
        df = df.drop("Label", axis=1)

    print(f"Total flows to analyze: {len(df)}")

    predictions, probabilities = predict(model, label_mapping, df, scaler, selector)

    harmful_count = sum(1 for pred in predictions if is_harmful(pred))
    benign_count = len(predictions) - harmful_count

    attack_types = {}
    for pred in predictions:
        if pred not in attack_types:
            attack_types[pred] = 0
        attack_types[pred] += 1

    print(f"\n{'=' * 70}")
    print("ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total flows analyzed:    {len(predictions)}")
    print(
        f"Benign flows:            {benign_count} ({benign_count / len(predictions) * 100:.2f}%)"
    )
    print(
        f"Harmful flows:           {harmful_count} ({harmful_count / len(predictions) * 100:.2f}%)"
    )

    if harmful_count > 0:
        print(f"\n{'=' * 70}")
        print("DETECTED THREATS")
        print(f"{'=' * 70}")
        for attack_type, count in sorted(attack_types.items()):
            if is_harmful(attack_type):
                percentage = count / len(predictions) * 100
                safe_attack_type = str(attack_type).encode("ascii", "replace").decode("ascii")
                print(f"  {safe_attack_type:30s} {count:6d} flows ({percentage:5.2f}%)")

    if verbose:
        print(f"\n{'=' * 70}")
        print("DETAILED FLOW ANALYSIS")
        print(f"{'=' * 70}")
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            confidence = max(proba) * 100
            status = "HARMFUL" if is_harmful(pred) else "BENIGN"
            safe_pred = str(pred).encode("ascii", "replace").decode("ascii")
            print(
                f"Flow {i + 1:5d}: {status:12s} | Type: {safe_pred:30s} | Confidence: {confidence:5.2f}%"
            )

    if output_file:
        results_df = df.copy()
        results_df["Predicted_Label"] = predictions
        results_df["Confidence"] = [max(proba) for proba in probabilities]
        results_df["Is_Harmful"] = [is_harmful(pred) for pred in predictions]

        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

    return harmful_count > 0


def analyze_single_flow(features_dict, model, label_mapping, scaler=None, selector=None):
    """
    Analyze a single network flow from command-line arguments.

    Args:
        features_dict: Dictionary of feature names and values
        model: Loaded XGBoost model
        label_mapping: Dictionary mapping class indices to labels
        scaler: Optional MinMaxScaler for preprocessing
        selector: Optional SelectKBest selector for feature selection
    """
    print(f"\n{'=' * 70}")
    print("Analyzing single network flow")
    print(f"{'=' * 70}\n")

    try:
        df = pd.DataFrame([features_dict])
    except Exception as e:
        print(f"Error creating flow data: {e}")
        sys.exit(1)

    predictions, probabilities = predict(model, label_mapping, df, scaler, selector)

    pred = predictions[0]
    confidence = max(probabilities[0]) * 100

    safe_pred = str(pred).encode("ascii", "replace").decode("ascii")
    print(f"Prediction:  {safe_pred}")
    print(f"Confidence:  {confidence:.2f}%")
    print(
        f"Status:      {'HARMFUL TRAFFIC DETECTED' if is_harmful(pred) else 'Benign traffic'}"
    )

    print("\nConfidence scores for all classes:")
    for class_name, prob in zip(label_mapping.values(), probabilities[0]):
        safe_class_name = str(class_name).encode("ascii", "replace").decode("ascii")
        print(f"  {safe_class_name:30s} {prob * 100:6.2f}%")

    return is_harmful(pred)


def main():
    parser = argparse.ArgumentParser(
        description="Intrusion Detection System - Analyze network traffic flows for harmful activity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  ids --file traffic_data.csv

  ids --file traffic_data.csv --verbose

  ids --file traffic_data.csv --output results.csv

  ids --json flow_features.json

  ids --list-classes
        """,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to CSV file containing traffic flows to analyze",
    )

    parser.add_argument(
        "--json",
        "-j",
        type=str,
        help="Path to JSON file containing a single flow's features",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information for each flow",
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Save detailed results to CSV file"
    )

    parser.add_argument(
        "--list-classes",
        "-l",
        action="store_true",
        help="List all detectable attack types and exit",
    )

    parser.add_argument(
        "--model-dir",
        "-m",
        type=str,
        default="models",
        help="Directory containing model artifacts (default: models)",
    )

    args = parser.parse_args()

    try:
        model, label_mapping, scaler, selector, preprocessing_config = load_model(args.model_dir)
    except FileNotFoundError as e:
        print(f"Error: Required model file not found - {e.filename}")
        print(f"Please ensure model artifacts exist in {args.model_dir}/")
        sys.exit(1)

    if args.list_classes:
        print("\nDetectable traffic classes:")
        print(f"{'=' * 70}")
        for idx, class_name in sorted(label_mapping.items(), key=lambda x: int(x[0])):
            status = "Benign" if class_name == "BENIGN" else "Harmful"

            safe_name = str(class_name).encode("ascii", "replace").decode("ascii")
            print(f"  [{idx:2s}] {safe_name:30s} ({status})")
        print(f"{'=' * 70}")
        print(f"\nTotal classes: {len(label_mapping)}")
        return 0

    if args.file:
        harmful_detected = analyze_traffic_file(
            args.file,
            model,
            label_mapping,
            scaler,
            selector,
            verbose=args.verbose,
            output_file=args.output,
        )
        return 1 if harmful_detected else 0

    if args.json:
        try:
            with open(args.json, "r") as f:
                features_dict = json.load(f)
            harmful_detected = analyze_single_flow(features_dict, model, label_mapping, scaler, selector)
            return 1 if harmful_detected else 0
        except FileNotFoundError:
            print(f"Error: JSON file not found - {args.json}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            sys.exit(1)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
