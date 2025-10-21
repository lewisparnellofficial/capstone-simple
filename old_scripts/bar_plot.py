import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_label_mapping(mapping_file="label_mapping.json"):
    """Load the label mapping from JSON file."""
    with open(mapping_file, "r") as f:
        label_map = json.load(f)

    return {int(k): v for k, v in label_map.items()}


def analyze_class_distribution(
    dataset_file="dataset.csv",
    mapping_file="label_mapping.json",
    save_plot=True,
    output_file="class_distribution.png",
):
    """
    Analyze and visualize class distribution in the CICIDS2017 dataset.

    Parameters:
    -----------
    dataset_file : str
        Path to the dataset CSV file
    mapping_file : str
        Path to the label mapping JSON file
    save_plot : bool
        Whether to save the plot to a file
    output_file : str
        Output filename for the plot
    """

    print("Loading dataset...")
    df = pd.read_csv(dataset_file)

    label_column = df.columns[-1]
    labels = df[label_column]

    print(f"Total samples: {len(labels):,}")

    label_map = load_label_mapping(mapping_file)

    class_counts = labels.value_counts().sort_index()

    total_samples = len(labels)
    class_percentages = class_counts / total_samples * 100

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"{'Class ID':<10} {'Class Name':<30} {'Count':<15} {'Percentage':<10}")
    print("-" * 70)

    for class_id in sorted(class_counts.index):
        class_name = label_map.get(class_id, f"Unknown_{class_id}")
        count = class_counts[class_id]
        percentage = class_percentages[class_id]
        print(f"{class_id:<10} {class_name:<30} {count:<15,} {percentage:>6.2f}%")

    print("-" * 70)
    print(f"{'Total':<10} {'':<30} {total_samples:<15,} {'100.00%':>6}")
    print("=" * 70)

    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class

    print("\nImbalance Metrics:")
    print(f"  - Largest class: {max_class:,} samples")
    print(f"  - Smallest class: {min_class:,} samples")
    print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"  - Number of classes: {len(class_counts)}")

    print("\nResampling Recommendation:")
    if imbalance_ratio > 100:
        print("  - SEVERE IMBALANCE DETECTED (>100:1)")
        print("  - Strong recommendation: Apply resampling techniques")
        print("  - Consider SMOTE, ADASYN, or class weights")
    elif imbalance_ratio > 10:
        print("  - MODERATE IMBALANCE DETECTED (>10:1)")
        print("  - Recommendation: Consider resampling or class weights")
    else:
        print("  - MILD IMBALANCE (<10:1)")
        print("  - Resampling may not be necessary")

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    class_names = [label_map.get(i, f"Unknown_{i}") for i in class_counts.index]

    ax1 = axes[0]
    colors = sns.color_palette("husl", len(class_counts))
    bars1 = ax1.bar(
        range(len(class_counts)), class_counts.values, color=colors, edgecolor="black"
    )
    ax1.set_xlabel("Attack Class", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax1.set_title(
        "CICIDS2017 Dataset - Class Distribution (Absolute Counts)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xticks(range(len(class_counts)))
    ax1.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    for i, (bar, count) in enumerate(zip(bars1, class_counts.values)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    ax1.text(
        0.98,
        0.97,
        f"Imbalance Ratio: {imbalance_ratio:.2f}:1",
        transform=ax1.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax2 = axes[1]
    bars2 = ax2.bar(
        range(len(class_counts)),
        class_percentages.values,
        color=colors,
        edgecolor="black",
    )
    ax2.set_xlabel("Attack Class", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Percentage of Total Samples (%)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "CICIDS2017 Dataset - Class Distribution (Percentage, Log Scale)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xticks(range(len(class_counts)))
    ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3, linestyle="--", which="both")

    for i, (bar, pct) in enumerate(zip(bars2, class_percentages.values)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    plt.tight_layout()

    if save_plot:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {output_file}")

    plt.show()

    return class_counts, class_percentages, imbalance_ratio


if __name__ == "__main__":
    class_counts, class_percentages, imbalance_ratio = analyze_class_distribution(
        dataset_file="dataset.csv",
        mapping_file="label_mapping.json",
        save_plot=True,
        output_file="class_distribution.png",
    )
