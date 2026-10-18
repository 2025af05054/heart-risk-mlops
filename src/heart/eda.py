"""Exploratory data analysis: generates the figures used in the notebook and report.

Usage:  python -m heart.eda
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart.config import (  # noqa: E402
    CATEGORICAL_FEATURES, BINARY_FEATURES, FIG_DIR, NUMERIC_FEATURES, TARGET,
)
from heart.data import load_clean  # noqa: E402

log = logging.getLogger(__name__)
PALETTE = {0: "#4c9f70", 1: "#c0392b"}
LABELS = {0: "No disease", 1: "Disease"}
NICE = {
    "age": "Age (years)", "trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate", "oldpeak": "ST depression", "ca": "Major vessels (0-3)",
    "sex": "Sex (1 = male)", "fbs": "Fasting sugar >120", "exang": "Exercise angina",
    "cp": "Chest pain type", "restecg": "Resting ECG", "slope": "ST slope", "thal": "Thalassemia",
}

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def class_balance(df, out=FIG_DIR):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / len(df):.0%})",
                ha="center", fontsize=10)
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "class_balance.png", out)


def missing_values(df, out=FIG_DIR):
    miss = df.isna().sum()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(miss.index, miss.values, color="#7f8c8d")
    ax.set_title("Missing values per column (after cleaning)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "missing_values.png", out)


def numeric_histograms(df, out=FIG_DIR):
    cols = [c for c in NUMERIC_FEATURES if c != "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, cols):
        for t in (0, 1):
            sns.histplot(df[df[TARGET] == t][col].dropna(), ax=ax, kde=True, stat="density",
                         color=PALETTE[t], label=LABELS[t], alpha=0.45, bins=20)
        ax.set_title(NICE[col])
        ax.set_xlabel("")
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution of numeric features by diagnosis", y=1.01, fontsize=13)
    return _save(fig, "numeric_histograms.png", out)


def numeric_boxplots(df, out=FIG_DIR):
    cols = [c for c in NUMERIC_FEATURES if c != "ca"]
    long = df.melt(id_vars=TARGET, value_vars=cols)
    long[TARGET] = long[TARGET].map(LABELS)
    g = sns.catplot(data=long, x=TARGET, y="value", col="variable", kind="box",
                    col_wrap=5, sharey=False, height=3, aspect=0.8,
                    palette=[PALETTE[0], PALETTE[1]], hue=TARGET, legend=False)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    path = out / "numeric_boxplots.png"
    g.figure.tight_layout()
    g.figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(g.figure)
    return path


def categorical_rates(df, out=FIG_DIR):
    cols = CATEGORICAL_FEATURES + BINARY_FEATURES + ["ca"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.flat, cols):
        rate = df.groupby(col)[TARGET].agg(["mean", "size"]).reset_index()
        ax.bar(rate[col].astype(int).astype(str), rate["mean"], color="#c0392b", alpha=0.8)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(NICE[col])
        ax.set_ylabel("Disease rate" if col in ("cp", "sex") else "")
    fig.suptitle("Heart disease rate per category", y=1.01, fontsize=13)
    return _save(fig, "categorical_rates.png", out)


def correlation_heatmap(df, out=FIG_DIR):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    for i in range(len(corr)):
        mask.iloc[i, i + 1:] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation matrix (Pearson)")
    return _save(fig, "correlation_heatmap.png", out)


def target_correlation(df, out=FIG_DIR):
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(corr.index, corr.values,
            color=["#c0392b" if v > 0 else "#4c9f70" for v in corr.values])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title("Correlation of each feature with the target")
    return _save(fig, "target_correlation.png", out)


def summary_table(df) -> pd.DataFrame:
    return df.describe().T.round(2)


def run_all(df=None, out: Path = FIG_DIR) -> list[Path]:
    df = load_clean() if df is None else df
    return [f(df, out) for f in (class_balance, missing_values, numeric_histograms,
                                 numeric_boxplots, categorical_rates, correlation_heatmap,
                                 target_correlation)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for p in run_all():
        log.info("wrote %s", p)
