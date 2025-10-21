import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def create_correlation_heatmap():
    """
    Create a correlation heatmap for continuous numeric features using Spearman correlation.
    Features:
    - Spearman correlation coefficient
    - Diverging color palette
    - Explicit color bar limits (-1 to +1)
    - Upper triangle masked (redundant)
    - Diagonal masked (no information)
    - Hierarchical clustering to reorder rows and columns
    - Cell annotations with two decimal places
    """

    df = pd.read_csv("dataset.csv")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Label" in numeric_cols:
        numeric_cols.remove("Label")

    numeric_data = df[numeric_cols]

    numeric_data = numeric_data.loc[:, numeric_data.std() > 0]

    corr_matrix = numeric_data.corr(method="spearman")

    corr_matrix = corr_matrix.fillna(0)

    dist_matrix = 1 - corr_matrix.values

    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=2.0, neginf=0.0)
    dist_matrix = np.clip(dist_matrix, 0, 2)

    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    dist_condensed = squareform(dist_matrix, checks=False)

    linkage = hierarchy.linkage(dist_condensed, method="average")
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    reorder_idx = dendro["leaves"]

    corr_matrix_reordered = corr_matrix.iloc[reorder_idx, reorder_idx]

    mask = np.triu(np.ones_like(corr_matrix_reordered, dtype=bool))

    plt.figure(figsize=(20, 18))

    sns.heatmap(
        corr_matrix_reordered,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman Correlation Coefficient"},
    )

    plt.title(
        "Spearman Correlation Heatmap of Continuous Features\n(Hierarchically Clustered)",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
    print("Correlation heatmap saved as 'correlation_heatmap.png'")

    print("\nCorrelation Matrix Statistics:")
    print(f"Number of features: {len(corr_matrix_reordered)}")
    print(f"Mean correlation: {corr_matrix_reordered.where(~mask).mean().mean():.3f}")
    print(
        f"Max correlation (excluding diagonal): {corr_matrix_reordered.where(~mask).max().max():.3f}"
    )
    print(
        f"Min correlation (excluding diagonal): {corr_matrix_reordered.where(~mask).min().min():.3f}"
    )

    plt.show()


if __name__ == "__main__":
    create_correlation_heatmap()
