import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numbers import Number
from scipy.stats import norm


def create_heatmap(df: pd.DataFrame, metrics, vmaxes: dict, group_by: str, value: Number, x: str, y: str, interpolated: bool = False):
    
    df_copy = df.copy()
    _, axs = plt.subplots(nrows=len(metrics), ncols=1, figsize=(5, 4 * len(metrics)))
    
    filtered_df = df_copy[df_copy[group_by] == value]
    for j, metric in enumerate(metrics):
        pivot_table = filtered_df.pivot_table(index=x, columns=y, values=f'{metric}_mean')
        data = pivot_table.values
        ax = axs[j]
        
        if interpolated:
            im = ax.imshow(
                data,
                cmap="magma", interpolation='bilinear',
                vmin=0.0, vmax=vmaxes[metric],
                aspect='auto', origin='upper'
            )
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_xticklabels(pivot_table.columns)
            ax.set_yticklabels(pivot_table.index)
            plt.colorbar(im, ax=ax)
        else:
            sns.heatmap(
                pivot_table, ax=ax,
                annot=True, fmt=".1f", cmap="magma",
                vmin=0., vmax=vmaxes[metric]
            )

        ax.set_title(f"{metric.capitalize()} ({group_by}={value})")
        ax.set_xlabel(y)
        ax.set_ylabel(x)
        ax.invert_yaxis()
    
    plt.tight_layout()


def create_plot(df: pd.DataFrame, metrics, group_by: str, degree: int = 2):
    plt.figure()
    sns.set_theme(style="whitegrid", rc={"figure.figsize":(10, 6)})
    colors = sns.color_palette('tab10', len(metrics))

    for metric, color in zip(metrics, colors):
        avg_group = (
            df.groupby(group_by)[[f'{metric}_mean', f'{metric}_std']]
            .mean()
            .reset_index()
        )

        plt.errorbar(
            avg_group[group_by],
            avg_group[f'{metric}_mean'],
            yerr=avg_group[f'{metric}_std'],
            color=color,
            fmt='o',
            capsize=5,
        )

        sns.regplot(
            data=avg_group,
            x=group_by,
            y=f'{metric}_mean',
            color=color,
            scatter=True,
            ci=False,
            order=degree,
            line_kws={'linestyle':'--'},
            label=f'{metric.capitalize()}'
        )

    plt.title(f"Metric Trends by {group_by}")
    plt.xlabel(group_by)
    plt.ylabel("Score")
    plt.xticks(avg_group[group_by])
    plt.legend(title="Metrics", loc='upper right')
    plt.tight_layout()


def create_histogram(df: pd.DataFrame, x: str, title: str, xlabel: str = None, ylabel: str = None, **kwargs):
    plt.figure()

    # Plot histogram and capture bin info
    hist = sns.histplot(df, x=x, **kwargs)
    bin_width = hist.patches[0].get_width()

    # Compute mean and std
    mean_val = df[x].mean()
    std_val = df[x].std()

    # Plot vertical line at mean
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

    # Prepare Gaussian fit scaled to histogram
    x_vals = np.linspace(df[x].min(), df[x].max(), 1000)
    y_vals = norm.pdf(x_vals, mean_val, std_val) * len(df[x]) * bin_width
    plt.plot(x_vals, y_vals, color='red', alpha=0.5, label=f'Std: {std_val:.2f}')

    # Labels and layout
    plt.legend()
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()


def plot_trend(df, x, y):
    sns.regplot(df, x=x, y=y, ci=None, line_kws={'linestyle':'--', 'color': 'black'}, order=2)


def create_scatter(
    df: pd.DataFrame, x: str, y: str,
    title: str = None, xlabel: str = None, ylabel: str = None, trend: bool = False,
    **kwargs
):
    plt.figure()
    sns.scatterplot(df, x=x, y=y, **kwargs)
    if trend:
        plot_trend(df, x=x, y=y)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()

