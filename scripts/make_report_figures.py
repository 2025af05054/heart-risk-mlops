"""Figures for the written report: architecture diagram + experiment summary from MLflow."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = "reports/figures"


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#2d3748", lw=1.1))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9.5,
            weight="bold", color="#1a202c")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=7.4,
            color="#2d3748", linespacing=1.25)


def arrow(ax, a, b, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.1,
                                 color="#4a5568", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center",
                fontsize=7, color="#4a5568", style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    # lanes
    for y0, h, label, c in [(4.55, 2.1, "Development & experimentation", "#f7fafc"),
                            (2.45, 1.95, "CI/CD (GitHub Actions)", "#f0fff4"),
                            (0.1, 2.2, "Serving & monitoring (Minikube)", "#ebf8ff")]:
        ax.add_patch(FancyBboxPatch((0.1, y0), 11.8, h, boxstyle="round,pad=0.01",
                                    fc=c, ec="#cbd5e0", lw=0.8))
        ax.text(0.25, y0 + h - 0.2, label, fontsize=8.5, color="#718096", weight="bold")
    Y1, Y2, Y3 = 4.85, 2.7, 0.45
    w, h = 1.75, 1.1
    box(ax, 0.4, Y1, w, h, "UCI repository", "processed.cleveland.data\n(mirror fallback)", "#fefcbf")
    box(ax, 2.6, Y1, w, h, "heart.data", "clean, binarise target,\nflag invalid -> NaN", "#e9d8fd")
    box(ax, 4.8, Y1, w, h, "EDA notebook", "heart.eda figures\nclass balance, corr", "#e9d8fd")
    box(ax, 7.0, Y1, w, h, "heart.train", "5 models x GridSearchCV\n5-fold CV, 1-SE rule", "#e9d8fd")
    box(ax, 9.4, Y1, 2.2, h, "MLflow", "params, metrics, plots,\nmodel registry (sqlite)", "#fed7e2")
    arrow(ax, (2.15, Y1 + .55), (2.6, Y1 + .55))
    arrow(ax, (4.35, Y1 + .55), (4.8, Y1 + .55))
    arrow(ax, (6.55, Y1 + .55), (7.0, Y1 + .55))
    arrow(ax, (8.75, Y1 + .55), (9.4, Y1 + .55), "log")

    box(ax, 0.4, Y2, w, h, "git push / PR", "GitHub repository", "#c6f6d5")
    box(ax, 2.6, Y2, w, h, "lint", "flake8", "#c6f6d5")
    box(ax, 4.8, Y2, w, h, "test", "pytest (32 tests)\nJUnit artifact", "#c6f6d5")
    box(ax, 7.0, Y2, w, h, "train + gate", "retrain, ROC-AUC >= 0.85\nmodel artifact", "#c6f6d5")
    box(ax, 9.4, Y2, 2.2, h, "docker", "build, smoke-test /predict\npush to GHCR", "#c6f6d5")
    for x in (2.15, 4.35, 6.55):
        arrow(ax, (x, Y2 + .55), (x + .45, Y2 + .55))
    arrow(ax, (8.75, Y2 + .55), (9.4, Y2 + .55))
    arrow(ax, (7.87, Y1), (7.87, Y2 + h), rad=0)

    box(ax, 0.4, Y3, w, h, "Client", "curl / app\nJSON request", "#bee3f8")
    box(ax, 2.6, Y3, w, h, "LB / Ingress", "LoadBalancer Svc :80\nheart-api.local", "#bee3f8")
    box(ax, 4.8, Y3, 2.2, h, "heart-api pods (x2, HPA)", "FastAPI + sklearn Pipeline\n/predict /health /metrics", "#90cdf4")
    box(ax, 7.6, Y3, w, h, "Prometheus", "pod discovery\nscrape /metrics 10s", "#feebc8")
    box(ax, 9.85, Y3, 1.75, h, "Grafana", "service + model\ndashboard", "#feebc8")
    arrow(ax, (2.15, Y3 + .55), (2.6, Y3 + .55))
    arrow(ax, (4.35, Y3 + .55), (4.8, Y3 + .55))
    arrow(ax, (7.6, Y3 + .55), (7.0, Y3 + .55), "scrape")
    arrow(ax, (9.35, Y3 + .55), (9.85, Y3 + .55))
    arrow(ax, (10.5, Y2), (5.9, Y3 + h), "image", rad=-0.15)
    fig.savefig(f"{OUT}/architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def experiment_summary():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(experiment_names=["heart-disease-classification"])
    cands = runs[runs["tags.mlflow.parentRunId"].notna()].copy()
    parents = runs[runs["tags.model_family"].notna()]
    id2name = dict(zip(parents["run_id"], parents["tags.model_family"]))
    cands["family"] = cands["tags.mlflow.parentRunId"].map(id2name)
    cands = cands.dropna(subset=["family"]).drop_duplicates(
        subset=["tags.mlflow.runName", "metrics.cv_roc_auc"])
    cmp = pd.read_csv("reports/model_comparison.csv").set_index("name")
    order = cmp.sort_values("cv_roc_auc").index

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    for i, fam in enumerate(order):
        vals = cands[cands.family == fam]["metrics.cv_roc_auc"]
        ax.scatter(vals, [i] * len(vals), alpha=0.45, s=22, color="#718096")
        ax.errorbar(cmp.loc[fam, "cv_roc_auc"], i, xerr=cmp.loc[fam, "cv_roc_auc_std"],
                    fmt="o", color="#c0392b", capsize=4, ms=7)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0.84, 0.96)
    n_low = int((cands["metrics.cv_roc_auc"] < 0.84).sum())
    if n_low:
        ax.set_ylim(-0.9, len(order) - 0.5)
        ax.text(0.842, -0.7, f"{n_low} candidate(s) < 0.84 not shown (e.g. strong L1, C=0.01:"
                " all coefficients shrunk to 0, AUC 0.5)", fontsize=7, color="#718096")
    ax.set_xlabel("5-fold CV ROC-AUC")
    ax.set_title("All grid candidates (grey) and best per family (red, +/- 1 SD)")
    top = cmp["cv_roc_auc"].max() - cmp.loc[cmp["cv_roc_auc"].idxmax(), "cv_roc_auc_std"]
    ax.axvline(top, ls="--", color="#2b6cb0", lw=1)
    ax.text(top, len(order) - 1.5, " 1-SE cutoff", color="#2b6cb0", fontsize=8)

    ax = axes[1]
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    data = cmp.loc[list(order)[::-1], metrics]
    data.columns = [m.replace("test_", "") for m in metrics]
    data.T.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")
    ax.set_ylim(0.6, 0.95)
    ax.set_title("Held-out test metrics (20%, n=61)")
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/experiment_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    experiment_summary()
    print("ok")
