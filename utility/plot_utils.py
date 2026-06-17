import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def distribution_plot(numeric_df, data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_cols = numeric_df.columns
    n = len(numeric_cols)
    cols_per_row = 4
    rows = math.ceil(n / cols_per_row)

    fig, axes = plt.subplots(
        rows, cols_per_row,
        figsize=(cols_per_row * 4, rows * 3.5)
    )
    axes = axes.flatten()

    palette = sns.color_palette("crest", n)

    for i, col in enumerate(numeric_cols):
        sns.histplot(
            numeric_df[col],   
            bins=20,
            kde=True,
            ax=axes[i],
            color=palette[i]
        )
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")

    for ax in axes[n:]:
        ax.remove()

    fig.suptitle(f"Distributions of {data} data", fontsize=16, y=1.02)
    fig.tight_layout()
    plt.show()
    
def corr_plot(corr_matrix):
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.6},
    )
    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.show()

def confusion_plot(confusion_matrix, model_name):
    plt.figure(figsize=(5, 4)) 
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds") 
    plt.title(f"Confusion Matrix | {model_name}") 
    plt.xlabel("Predicted")  
    plt.ylabel("True") 
    plt.tight_layout() 
    plt.show() 

def plot_roc_curve(
    y_true,
    y_score,
    model_name = "Model",
    figsize = (7, 6),
    color = "navy",
    show_thresholds = False,
    save_path = None,
):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize = figsize)

    ax.plot(
        fpr,
        tpr,
        color = color,
        lw = 2.5,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        lw=1.5,
        label="Random Classifier"
    )

    if show_thresholds:
        n_points = min(5, len(thresholds))
        idx = np.linspace(0, len(thresholds) - 1, n_points, dtype=int)

        for i in idx:
            ax.scatter(fpr[i], tpr[i], s=40, color=color)
            ax.annotate(
                f"{thresholds[i]:.2f}",
                (fpr[i], tpr[i]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points"
            )

    ax.set_title(
        f"Receiver Operating Characteristic (ROC)\n{model_name}",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax, roc_auc

import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(
    data,
    target_col,
    labels = None,
    colors = None,
    figsize = (12, 5),
    title = None,
    show_percentages = True,
    save_path = None,
):
    
    counts = data[target_col].value_counts().sort_index()

    if labels is None:
        labels = [str(x) for x in counts.index]

    if colors is None:
        colors = plt.cm.Set2.colors[:len(counts)]

    fig, axes = plt.subplots(
        1,
        2,
        figsize = figsize,
        gridspec_kw = {"width_ratios": [1.2, 1]}
    )
    
    bars = axes[0].bar(
        labels,
        counts.values,
        color = colors,
        edgecolor = "black",
        linewidth = 1.0
    )

    axes[0].set_title(
        "Class Distribution",
        fontsize = 13,
        fontweight = "bold"
    )

    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    total = counts.sum()

    for bar, count in zip(bars, counts.values):

        percentage = 100 * count / total

        text = (
            f"{count:,}\n({percentage:.1f}%)"
            if show_percentages
            else f"{count:,}"
        )

        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha = "center",
            va = "bottom",
            fontsize = 10,
            fontweight = "bold"
        )

    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels = labels,
        colors = colors,
        autopct = "%1.1f%%",
        startangle = 90,
        counterclock = False,
        wedgeprops = {
            "edgecolor": "white",
            "linewidth": 2
        },
        textprops={
            "fontsize": 10
        }
    )

    for autotext in autotexts:
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    axes[1].set_title(
        "Class Proportion",
        fontsize = 13,
        fontweight = "bold"
    )

    if title is None:
        title = f"Distribution of '{target_col}'"

    fig.suptitle(
        title,
        fontsize = 16,
        fontweight = "bold",
        y = 1.02
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi = 300,
            bbox_inches = "tight"
        )

    plt.show()

    return fig, axes