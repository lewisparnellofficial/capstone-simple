"""
Violin Plot Visualization for Network Traffic Attack Type Feature Distribution

This script creates a single violin plot comparing the distribution of the top features
(based on model feature importance) across different network traffic classes.
These visualizations help understand how the model learns different attack signatures.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

warnings.filterwarnings("ignore")


def load_model_and_get_top_features(model_path="model.json", top_n=5):
    """
    Load the trained XGBoost model and extract top N most important features.

    Args:
        model_path: Path to the saved model JSON file
        top_n: Number of top features to extract

    Returns:
        List of top feature names and their importance scores
    """
    print(f"Loading model from {model_path}...")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    df = pd.read_csv("dataset.csv")
    feature_names = [col for col in df.columns if col != "Label"]

    try:
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance_scores))
        print("Using feature_importances_ attribute")
    except AttributeError:
        print("Using get_score() method...")
        importance_dict = model.get_booster().get_score(importance_type="weight")
        feature_importance = {}
        for idx, fname in enumerate(feature_names):
            fkey = f"f{idx}"
            if fkey in importance_dict:
                feature_importance[fname] = importance_dict[fkey]

    if not feature_importance:
        print("Warning: Could not extract feature importance from model.")
        print("Falling back to predefined important features...")

        important_features = [
            "Flow Duration",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Fwd Packet Length Mean",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Bwd Packet Length Mean",
            "Flow IAT Mean",
        ]
        top_features = [f for f in important_features if f in feature_names][:top_n]
        sorted_features = [(f, 0.0) for f in top_features]
    else:
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        top_features = [f for f, _ in sorted_features[:top_n]]

    print(f"\nTop {top_n} most important features:")
    for i, (feature, score) in enumerate(sorted_features[:top_n], 1):
        print(f"  {i}. {feature}: {score:.2f}")

    return top_features, sorted_features[:top_n]


def load_and_sample_data(csv_path, sample_size=50000, random_state=42):
    """
    Load dataset and create a stratified sample for visualization.

    Args:
        csv_path: Path to the dataset CSV file
        sample_size: Number of samples to use (for performance)
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    print(f"\nLoading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Total samples: {len(df):,}")
    print(f"Number of features: {len(df.columns) - 1}")

    print(f"Creating stratified sample of {sample_size:,} instances...")
    df_sample = (
        df.groupby("Label", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), max(100, int(sample_size * len(x) / len(df)))),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )

    print(f"Sample size: {len(df_sample):,}")

    return df_sample


def create_violin_plot(
    df,
    features,
    feature_scores,
    output_path="violin_plot_top_features.png",
    top_n_classes=8,
):
    """
    Create a single violin plot with subplots for top features.

    Args:
        df: DataFrame with network traffic data
        features: List of top feature names to plot
        feature_scores: List of tuples (feature_name, importance_score)
        output_path: Path to save the plot
        top_n_classes: Number of most common classes to include
    """
    top_classes = df["Label"].value_counts().head(top_n_classes).index.tolist()
    df_filtered = df[df["Label"].isin(top_classes)].copy()

    print(f"\nCreating violin plot for top {len(features)} features...")
    print(f"Including top {top_n_classes} traffic classes")

    sns.set_style("whitegrid")

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4 * n_features))

    if n_features == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", n_colors=top_n_classes)

    for idx, (feature, (feat_name, score)) in enumerate(zip(features, feature_scores)):
        ax = axes[idx]

        feature_data = df_filtered[[feature, "Label"]].copy()
        feature_data = feature_data[np.isfinite(feature_data[feature])]

        Q1 = feature_data[feature].quantile(0.25)
        Q3 = feature_data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        feature_data = feature_data[
            (feature_data[feature] >= lower_bound)
            & (feature_data[feature] <= upper_bound)
        ]

        try:
            sns.violinplot(
                data=feature_data,
                x="Label",
                y=feature,
                palette=colors,
                ax=ax,
                cut=0,
                inner="box",
                linewidth=1.5,
                saturation=0.8,
            )

            ax.set_xlabel("Traffic Class", fontsize=11, fontweight="bold")
            ax.set_ylabel(feature, fontsize=11, fontweight="bold")

            ax.set_title(
                f"Feature Importance: {score:.0f}",
                fontsize=10,
                loc="right",
                color="gray",
                style="italic",
            )

            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=9
            )

            ax.yaxis.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

        except Exception as e:
            print(f"    Error plotting {feature}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error plotting {feature}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    plt.suptitle(
        "Attack Type Feature Distribution - Top Features by Importance\n"
        + "Violin plots showing how key features distinguish between benign and malicious traffic",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nViolin plot saved to {output_path}")
    plt.close()


def create_summary_stats(df, features, top_n_classes=8):
    """
    Create a summary table of statistics for top features across attack types.

    Args:
        df: DataFrame with network traffic data
        features: List of feature names
        top_n_classes: Number of top classes to include
    """
    print("\n" + "=" * 80)
    print("Feature Distribution Summary Statistics")
    print("=" * 80)

    top_classes = df["Label"].value_counts().head(top_n_classes).index.tolist()
    df_filtered = df[df["Label"].isin(top_classes)].copy()

    for feature in features:
        print(f"\n{feature}:")
        print("-" * 80)

        stats = df_filtered.groupby("Label")[feature].agg(
            [
                ("count", "count"),
                ("mean", "mean"),
                ("std", "std"),
                ("min", "min"),
                ("25%", lambda x: x.quantile(0.25)),
                ("50%", lambda x: x.quantile(0.50)),
                ("75%", lambda x: x.quantile(0.75)),
                ("max", "max"),
            ]
        )

        print(stats.to_string())


def main():
    """Main execution function."""
    print("=" * 80)
    print("Network Traffic Attack Type Feature Distribution Analysis")
    print("Top Features Violin Plot Visualization")
    print("=" * 80)

    dataset_path = "dataset.csv"
    model_path = "model.json"
    output_file = "violin_plot_top_features.png"
    top_n_features = 5
    sample_size = 50000
    top_n_classes = 8

    try:
        top_features, feature_scores = load_model_and_get_top_features(
            model_path, top_n=top_n_features
        )

        df = load_and_sample_data(dataset_path, sample_size=sample_size)

        create_violin_plot(
            df,
            top_features,
            feature_scores,
            output_path=output_file,
            top_n_classes=top_n_classes,
        )

        create_summary_stats(df, top_features, top_n_classes=top_n_classes)

        print("\n" + "=" * 80)
        print("Visualization Complete!")
        print("=" * 80)
        print(f"\nViolin plot saved to: {output_file}")
        print("\nThis visualization shows:")
        print("  • Distribution of the most important features for attack detection")
        print(
            "  • How different attack types exhibit distinct patterns in key features"
        )
        print("  • The variability and density of features within each traffic class")
        print("  • Which features the model relies on most to distinguish attacks")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the following files exist:")
        print(f"  - {model_path} (trained model)")
        print(f"  - {dataset_path} (dataset)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
