import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_violin(plot_df, label_col="label", diff_col="D"):
    categories = ["free", "confined", "bound"]

    # Keep only rows belonging to these three categories
    df_filtered = plot_df[plot_df[label_col].isin(categories)].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(
        data=df_filtered,
        x=label_col,
        y=diff_col,
        order=categories,   # ensures consistent left-to-right ordering
        ax=ax,
        palette="Set2"
    )

    ax.set_xlabel("Motion Class")
    ax.set_ylabel("Diffusion Coefficient")
    ax.set_title("Diffusion Coefficient by Motion Class")

    plt.tight_layout()
    plt.show()
    return fig


def plot_histogram(plot_df, label_col="label", diff_col="D", bins=60):
    categories = ["free", "confined", "bound"]
    colors = {"free": "tab:blue", "confined": "tab:green", "bound": "tab:red"}

    fig, ax = plt.subplots(figsize=(9, 6))

    # Overall histogram (all data, density-normalized so it's on the same
    # scale as the KDE curves)
    ax.hist(
        plot_df[diff_col].dropna(),
        bins=bins,
        density=True,
        alpha=0.3,
        color="grey",
        edgecolor="black",
        linewidth=0.3,
        label="Overall"
    )

    # KDE per label, overlaid
    for cat in categories:
        subset = plot_df.loc[plot_df[label_col] == cat, diff_col].dropna()
        if len(subset) > 1:  # KDE needs at least 2 points
            sns.kdeplot(
                subset,
                ax=ax,
                color=colors[cat],
                linewidth=2,
                label=cat,
                bw_adjust=1.2,
                clip=(0, None)
            )

            mean_val = subset.mean()
            ax.axvline(
                mean_val,
                color=colors[cat],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"mean D* for {cat} samples = {mean_val:.3g}m^2/s"
            )


    ax.set_xlabel("Diffusion Coefficient")
    ax.set_ylabel("Density")
    ax.set_title("Diffusion Coefficient: Overall Histogram with KDE by label provided by model")
    ax.legend(title="")

    plt.tight_layout()
    plt.show()
    return fig

def map_frames(labels_df, difc_df):  # Check notion (github?) for hoe to cut each sample

    labels_df = labels_df.iloc[1:].reset_index(drop=True)
    labels_df["Frame"] = labels_df["Frame"].astype(int) + 1

    print(labels_df.head())
    print(difc_df.head())

    plot_df = pd.merge(labels_df, difc_df, on="Frame", how="outer")

    return plot_df



if __name__ == "__main__":
    #54
    labels_df = pd.read_csv(r"H:\summer_internship\analysis\label_per_frame\54_per_frame_label.csv", names = ["Frame", "Label"])
    difc_df = pd.read_csv(r"H:\summer_internship\analysis\locoli_tracking_data\cell_54_locoli.csv", names = ["X_pos", "Y_pos", "Frame", "Track_ID", "Difc"])
    plot_df_54 = map_frames(labels_df, difc_df)

    plot_df_54 = plot_df_54.iloc[884:].copy()

    #57
    labels_df = pd.read_csv(r"H:\summer_internship\analysis\label_per_frame\58_per_frame_label.csv", names = ["Frame", "Label"])
    difc_df = pd.read_csv(r"H:\summer_internship\analysis\locoli_tracking_data\cell_58_locoli.csv", names = ["X_pos", "Y_pos", "Frame", "Track_ID", "Difc"])
    plot_df_57 = map_frames(labels_df, difc_df)

    plot_df_57 = plot_df_57.iloc[:1464].copy()

    #66
    labels_df = pd.read_csv(r"H:\summer_internship\analysis\label_per_frame\66_per_frame_label.csv", names = ["Frame", "Label"])
    difc_df = pd.read_csv(r"H:\summer_internship\analysis\locoli_tracking_data\cell_66_locoli.csv", names = ["X_pos", "Y_pos", "Frame", "Track_ID", "Difc"])
    plot_df_66 = map_frames(labels_df, difc_df)

    plot_df_66 = plot_df_66.iloc[:516].copy()

    #69
    labels_df = pd.read_csv(r"H:\summer_internship\analysis\label_per_frame\69_per_frame_label.csv", names = ["Frame", "Label"])
    difc_df = pd.read_csv(r"H:\summer_internship\analysis\locoli_tracking_data\cell_69_locoli.csv", names = ["X_pos", "Y_pos", "Frame", "Track_ID", "Difc"])
    plot_df_69 = map_frames(labels_df, difc_df)

    plot_df_69 = plot_df_69.iloc[:370].copy()

    #74
    labels_df = pd.read_csv(r"H:\summer_internship\analysis\label_per_frame\74_per_frame_label.csv", names = ["Frame", "Label"])
    difc_df = pd.read_csv(r"H:\summer_internship\analysis\locoli_tracking_data\cell_74_locoli.csv", names = ["X_pos", "Y_pos", "Frame", "Track_ID", "Difc"])
    plot_df_74 = map_frames(labels_df, difc_df)

    plot_df_74 = plot_df_74.iloc[:775].copy()

    #87
    labels_df = pd.read_csv(r"H:\summer_internship\analysis\label_per_frame\87_per_frame_label.csv", names = ["Frame", "Label"])
    difc_df = pd.read_csv(r"H:\summer_internship\analysis\locoli_tracking_data\cell_87_locoli.csv", names = ["X_pos", "Y_pos", "Frame", "Track_ID", "Difc"])
    plot_df_87 = map_frames(labels_df, difc_df)

    plot_df_87 = plot_df_87.iloc[:814].copy()
    
    


# stack all plots
    plot_df = pd.concat([plot_df_54, plot_df_57, plot_df_66, plot_df_69])
    plot_df = plot_df.dropna(subset=["Difc"])
    print(plot_df.head(10))


    plot_histogram(plot_df, label_col="Label", diff_col="Difc")