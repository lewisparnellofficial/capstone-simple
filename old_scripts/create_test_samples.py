#!/usr/bin/env python3
"""
Test Sample Generator for IDS

Creates representative samples from the full dataset with configurable
traffic distributions for testing the IDS under different scenarios.
"""

import argparse
import json
import sys

import pandas as pd


def load_dataset(dataset_path="dataset.csv"):
    """Load the full dataset."""
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} total flows")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def show_dataset_distribution(df):
    """Display the distribution of traffic types in the dataset."""
    print("\n" + "="*70)
    print("DATASET DISTRIBUTION")
    print("="*70)

    if "Label" not in df.columns:
        print("Error: 'Label' column not found in dataset")
        return

    label_counts = df["Label"].value_counts()
    total = len(df)

    for label, count in label_counts.items():
        percentage = count / total * 100
        # Handle potential encoding issues in labels
        safe_label = str(label).encode('ascii', 'replace').decode('ascii')
        print(f"  {safe_label:30s} {count:10d} flows ({percentage:6.2f}%)")

    print("="*70)
    print(f"  {'TOTAL':30s} {total:10d} flows")
    print()


def create_sample(df, scenario_config, output_file):
    """
    Create a sample dataset based on scenario configuration.

    Args:
        df: Full dataset DataFrame
        scenario_config: Dictionary mapping labels to desired counts or percentages
        output_file: Path to save the sample
    """
    if "Label" not in df.columns:
        print("Error: 'Label' column not found in dataset")
        sys.exit(1)

    available_labels = df["Label"].unique()
    samples = []

    print("\n" + "="*70)
    print("CREATING SAMPLE")
    print("="*70)

    total_requested = 0

    for label, amount in scenario_config.items():
        if label not in available_labels:
            print(f"Warning: Label '{label}' not found in dataset, skipping")
            continue

        label_df = df[df["Label"] == label]
        available_count = len(label_df)

        # If amount is between 0 and 1, treat as percentage
        if 0 < amount < 1:
            sample_count = int(amount * len(df))
            print(f"  {label:30s} {amount*100:5.1f}% = {sample_count} flows", end="")
        else:
            sample_count = int(amount)
            print(f"  {label:30s} {sample_count} flows", end="")

        # Check if we have enough samples
        if sample_count > available_count:
            print(f" (only {available_count} available, using all)")
            sample_count = available_count
        else:
            print()

        # Sample randomly from this label
        if sample_count > 0:
            label_sample = label_df.sample(n=sample_count, random_state=42)
            samples.append(label_sample)
            total_requested += sample_count

    if not samples:
        print("Error: No valid samples could be created")
        sys.exit(1)

    # Combine all samples and shuffle
    result_df = pd.concat(samples, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to file
    result_df.to_csv(output_file, index=False)

    print("="*70)
    print(f"Total flows in sample: {len(result_df)}")
    print(f"Saved to: {output_file}")
    print()

    # Show actual distribution
    print("="*70)
    print("ACTUAL SAMPLE DISTRIBUTION")
    print("="*70)
    label_counts = result_df["Label"].value_counts()
    for label, count in label_counts.items():
        percentage = count / len(result_df) * 100
        # Handle potential encoding issues in labels
        safe_label = str(label).encode('ascii', 'replace').decode('ascii')
        print(f"  {safe_label:30s} {count:6d} flows ({percentage:5.2f}%)")
    print("="*70)


def create_scenario(df, scenario_name, output_file):
    """Create a predefined scenario."""
    scenarios = {
        "mostly_benign": {
            "description": "95% benign traffic with scattered attacks",
            "config": {
                "BENIGN": 0.95,
                "DDoS": 0.02,
                "PortScan": 0.015,
                "Bot": 0.01,
                "DoS Hulk": 0.005
            }
        },
        "under_attack": {
            "description": "Heavy attack scenario - 50% attacks, 50% benign",
            "config": {
                "BENIGN": 0.50,
                "DDoS": 0.15,
                "DoS Hulk": 0.10,
                "DoS GoldenEye": 0.08,
                "PortScan": 0.07,
                "Bot": 0.05,
                "DoS slowloris": 0.03,
                "FTP-Patator": 0.02
            }
        },
        "mixed_attacks": {
            "description": "90% benign with diverse attack types",
            "config": {
                "BENIGN": 0.90,
                "DDoS": 0.02,
                "DoS Hulk": 0.015,
                "PortScan": 0.015,
                "Bot": 0.01,
                "SSH-Patator": 0.01,
                "FTP-Patator": 0.01,
                "DoS GoldenEye": 0.01,
                "DoS slowloris": 0.005,
                "Web Attack – Brute Force": 0.005
            }
        },
        "all_attack_types": {
            "description": "Small sample with ALL attack types represented",
            "config": {
                "BENIGN": 500,
                "Bot": 50,
                "DDoS": 50,
                "DoS GoldenEye": 50,
                "DoS Hulk": 50,
                "DoS Slowhttptest": 50,
                "DoS slowloris": 50,
                "FTP-Patator": 50,
                "Heartbleed": 50,
                "Infiltration": 50,
                "PortScan": 50,
                "SSH-Patator": 50,
                "Web Attack – Brute Force": 50,
                "Web Attack – Sql Injection": 50,
                "Web Attack – XSS": 50
            }
        },
        "web_attacks": {
            "description": "Focus on web-based attacks",
            "config": {
                "BENIGN": 0.85,
                "Web Attack – Brute Force": 0.05,
                "Web Attack – Sql Injection": 0.05,
                "Web Attack – XSS": 0.05
            }
        },
        "dos_focused": {
            "description": "Various DoS attack types",
            "config": {
                "BENIGN": 0.70,
                "DoS Hulk": 0.10,
                "DoS GoldenEye": 0.08,
                "DoS slowloris": 0.07,
                "DoS Slowhttptest": 0.05
            }
        },
        "small_test": {
            "description": "Small sample for quick testing (100 flows)",
            "config": {
                "BENIGN": 80,
                "DDoS": 10,
                "PortScan": 5,
                "Bot": 5
            }
        }
    }

    if scenario_name not in scenarios:
        print(f"Error: Unknown scenario '{scenario_name}'")
        print(f"\nAvailable scenarios:")
        for name, info in scenarios.items():
            print(f"  {name:20s} - {info['description']}")
        sys.exit(1)

    scenario = scenarios[scenario_name]
    print(f"\nScenario: {scenario_name}")
    print(f"Description: {scenario['description']}")

    create_sample(df, scenario["config"], output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Create test samples from the full dataset with configurable traffic distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset distribution
  python create_test_samples.py --show-distribution

  # Create a predefined scenario
  python create_test_samples.py --scenario mostly_benign --output benign_sample.csv

  # List all available scenarios
  python create_test_samples.py --list-scenarios

  # Create custom sample from JSON config
  python create_test_samples.py --config my_config.json --output custom_sample.csv

Example JSON config file:
{
  "BENIGN": 1000,
  "DDoS": 100,
  "PortScan": 50,
  "Bot": 50
}

Or use percentages (values between 0 and 1):
{
  "BENIGN": 0.95,
  "DDoS": 0.03,
  "PortScan": 0.02
}
        """
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="dataset.csv",
        help="Path to the full dataset (default: dataset.csv)"
    )

    parser.add_argument(
        "--show-distribution",
        action="store_true",
        help="Show the distribution of traffic types in the dataset"
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available predefined scenarios"
    )

    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        help="Use a predefined scenario (use --list-scenarios to see options)"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to JSON config file with custom traffic distribution"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for the sample (required with --scenario or --config)"
    )

    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.dataset)

    # Show distribution mode
    if args.show_distribution:
        show_dataset_distribution(df)
        return 0

    # List scenarios mode
    if args.list_scenarios:
        scenarios = {
            "mostly_benign": "95% benign traffic with scattered attacks",
            "under_attack": "Heavy attack scenario - 50% attacks, 50% benign",
            "mixed_attacks": "90% benign with diverse attack types",
            "all_attack_types": "Small sample with ALL attack types represented",
            "web_attacks": "Focus on web-based attacks",
            "dos_focused": "Various DoS attack types",
            "small_test": "Small sample for quick testing (100 flows)"
        }
        print("\nAvailable scenarios:")
        print("="*70)
        for name, description in scenarios.items():
            print(f"  {name:20s} - {description}")
        print("="*70)
        return 0

    # Create sample from scenario
    if args.scenario:
        if not args.output:
            print("Error: --output is required when using --scenario")
            sys.exit(1)
        create_scenario(df, args.scenario, args.output)
        return 0

    # Create sample from custom config
    if args.config:
        if not args.output:
            print("Error: --output is required when using --config")
            sys.exit(1)

        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            create_sample(df, config, args.output)
            return 0
        except FileNotFoundError:
            print(f"Error: Config file not found - {args.config}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file - {e}")
            sys.exit(1)

    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
