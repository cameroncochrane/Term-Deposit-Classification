from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from deposit_classification.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    X_train_path: Path = PROCESSED_DATA_DIR / "X_train.pkl",
    y_train_path: Path = PROCESSED_DATA_DIR / "y_train.pkl",
    output_directory: Path = FIGURES_DIR
):
    print("plots")

    with open(X_train_path, "rb") as f:
        X_train = pickle.load(f)

    with open(y_train_path, "rb") as f:
        y_train = pickle.load(f)

    eda_data = pd.concat([X_train.copy(), y_train.copy()], axis=1, join="inner")


    # Unique value diversity assessment for all columns except age, balance, campaign, and duration (these can be considered continuous):
    exclude_cols = ['age', 'balance', 'duration','campaign']
    categorical_cols = [col for col in eda_data.columns if col not in exclude_cols]
    print("Unique value assessment: No. unique values per selected column")
    for col in categorical_cols:
        print(f"{col}: {eda_data[col].nunique()}")
    

    # Value distribution plots
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        value_counts = eda_data[col].value_counts()
        axes[i].bar(value_counts.index, value_counts.values)
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    fig.suptitle("Categorical (Non-Continuous) Value Counts", fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_directory / "categorical_value_counts.png", dpi=300, bbox_inches='tight')
    plt.close()


    # Pearson correlation:
    pearson_cols = ['age', 'balance', 'duration','campaign','day','y']
    y_numeric = eda_data['y']
    pearson_df = eda_data[pearson_cols[:-1]].copy()
    pearson_df['y'] = y_numeric
    pearson_with_y = pearson_df.corr(method='pearson')['y'].drop('y')
    print("Pearson correlation coefficients with y:")
    print(pearson_with_y)


    # Outlier detection via plotting of numeric columns vs y:
    boxplot_cols = ['age', 'balance', 'duration','campaign','y']
    num_cols = [col for col in boxplot_cols if col != 'y']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, num_cols):
        sns.boxplot(data=eda_data, x='y', y=col, ax=ax)
        ax.set_title(f'{col} vs y')
        ax.set_xlabel('y')
        ax.set_ylabel(col)
    for ax in axes[len(num_cols):]:
        ax.axis('off')
    plt.tight_layout()    
    plt.savefig(output_directory / "numeric_vs_target_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Boxplots for numeric columns only (exclude target 'y')
    num_cols = [c for c in boxplot_cols if c != 'y']
    n_cols = 2
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, num_cols):
        sns.boxplot(y=eda_data[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        ax.set_ylabel(col)

    for ax in axes[len(num_cols):]:
        ax.axis('off')
    plt.tight_layout()    
    plt.savefig(output_directory / "numeric_columns_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    app()
