"""
Intrusion Detection System (IDS) - Command Line Interface

This CLI tool uses a trained XGBoost model to detect harmful traffic in network flows.
"""

import argparse
import json
import sys

import pandas as pd
import xgboost as xgb


def load_model():
    """Load the best hyperparameters, the trained model, and label mapping."""
    try:
        with open("best_hyperparameters.json", "r") as f:
            hyperparameters = json.load(f)

        model = xgb.XGBClassifier(**hyperparameters)
        model.load_model("model.json")

        with open("label_mapping.json", "r") as f:
            label_mapping = json.load(f)

        return model, label_mapping
    except FileNotFoundError as e:
        print(f"Error: Required model file not found - {e.filename}")
        print(
            "Please ensure model.json, best_hyperparameters.json, and label_mapping.json exist."
        )
        sys.exit(1)


def predict(model, label_mapping, features):
    """
    Make predictions on new data.

    Args:
        model: Loaded XGBoost model
        label_mapping: Dictionary mapping class indices to labels
        features: DataFrame or numpy array with the same features as training data

    Returns:
        predictions: Original label names
        probabilities: Probability scores for each class
    """
    y_pred_indices = model.predict(features)
    y_pred_proba = model.predict_proba(features)

    predictions = [label_mapping[str(int(idx))] for idx in y_pred_indices]

    return predictions, y_pred_proba


def is_harmful(label):
    """Determine if a traffic label indicates harmful activity."""
    return label.upper() != "BENIGN"


def analyze_traffic_file(
    file_path, model, label_mapping, verbose=False, output_file=None
):
    """
    Analyze a CSV file containing network traffic flows.

    Args:
        file_path: Path to CSV file with traffic data
        model: Loaded XGBoost model
        label_mapping: Dictionary mapping class indices to labels
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

    predictions, probabilities = predict(model, label_mapping, df)

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
                print(f"  {attack_type:30s} {count:6d} flows ({percentage:5.2f}%)")

    if verbose:
        print(f"\n{'=' * 70}")
        print("DETAILED FLOW ANALYSIS")
        print(f"{'=' * 70}")
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            confidence = max(proba) * 100
            status = "=� HARMFUL" if is_harmful(pred) else " BENIGN"
            print(
                f"Flow {i + 1:5d}: {status:12s} | Type: {pred:30s} | Confidence: {confidence:5.2f}%"
            )

    if output_file:
        results_df = df.copy()
        results_df["Predicted_Label"] = predictions
        results_df["Confidence"] = [max(proba) for proba in probabilities]
        results_df["Is_Harmful"] = [is_harmful(pred) for pred in predictions]

        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

    return harmful_count > 0


def analyze_single_flow(features_dict, model, label_mapping):
    """
    Analyze a single network flow from command-line arguments.

    Args:
        features_dict: Dictionary of feature names and values
        model: Loaded XGBoost model
        label_mapping: Dictionary mapping class indices to labels
    """
    print(f"\n{'=' * 70}")
    print("Analyzing single network flow")
    print(f"{'=' * 70}\n")

    try:
        df = pd.DataFrame([features_dict])
    except Exception as e:
        print(f"Error creating flow data: {e}")
        sys.exit(1)

    predictions, probabilities = predict(model, label_mapping, df)

    pred = predictions[0]
    confidence = max(probabilities[0]) * 100

    print(f"Prediction:  {pred}")
    print(f"Confidence:  {confidence:.2f}%")
    print(
        f"Status:      {'=� HARMFUL TRAFFIC DETECTED' if is_harmful(pred) else ' Benign traffic'}"
    )

    print("\nConfidence scores for all classes:")
    for class_name, prob in zip(label_mapping.values(), probabilities[0]):
        print(f"  {class_name:30s} {prob * 100:6.2f}%")

    return is_harmful(pred)


def main():
    parser = argparse.ArgumentParser(
        description="Intrusion Detection System - Analyze network traffic flows for harmful activity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python ids.py --file traffic_data.csv

  python ids.py --file traffic_data.csv --verbose

  python ids.py --file traffic_data.csv --output results.csv

  python ids.py --json flow_features.json

  python ids.py --list-classes
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

    args = parser.parse_args()

    model, label_mapping = load_model()

    if args.list_classes:
        print("\nDetectable traffic classes:")
        print(f"{'=' * 70}")
        for idx, class_name in sorted(label_mapping.items(), key=lambda x: int(x[0])):
            status = "Benign" if class_name == "BENIGN" else "Harmful"
            print(f"  [{idx:2s}] {class_name:30s} ({status})")
        print(f"{'=' * 70}")
        print(f"\nTotal classes: {len(label_mapping)}")
        return 0

    if args.file:
        harmful_detected = analyze_traffic_file(
            args.file,
            model,
            label_mapping,
            verbose=args.verbose,
            output_file=args.output,
        )
        return 1 if harmful_detected else 0

    if args.json:
        try:
            with open(args.json, "r") as f:
                features_dict = json.load(f)
            harmful_detected = analyze_single_flow(features_dict, model, label_mapping)
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
