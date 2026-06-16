import math
import matplotlib.pyplot as plt
import seaborn as sns

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

