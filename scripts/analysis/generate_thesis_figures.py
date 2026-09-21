from __future__ import annotations

import csv
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "official_test"
OUT_DIR = ROOT / "figures" / "official"

SUMMARY_PATH = DATA_DIR / "summary_statistics_and_confidence_intervals.tsv"
RUNS_PATH = DATA_DIR / "per_run_metrics.tsv"

DATASETS = ["FB15k-237", "WN18RR", "CoDEx-M"]
MODELS = ["TransE", "DistMult"]
UNITS = [(dataset, model) for dataset in DATASETS for model in MODELS]
CONTRASTS = ["correct_text_vs_random", "correct_text_vs_shuffled_text"]
CONTRAST_LABELS = {
    "correct_text_vs_random": "vs random",
    "correct_text_vs_shuffled_text": "vs shuffled text",
}
METRICS = [
    "combined_filtered_test_mrr",
    "calibrated_test_multiclass_nll",
    "calibrated_test_eaurc",
]
METRIC_TITLES = {
    "combined_filtered_test_mrr": "Filtered MRR",
    "calibrated_test_multiclass_nll": "Calibrated NLL",
    "calibrated_test_eaurc": "Calibrated E-AURC",
}
CONDITIONS = ["random", "correct_text", "shuffled_text"]
CONDITION_LABELS = {
    "random": "Random",
    "correct_text": "Correct text",
    "shuffled_text": "Shuffled text",
}
CONDITION_COLORS = {
    "random": "#6B7280",
    "correct_text": "#2563EB",
    "shuffled_text": "#D97706",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def unit_label(unit: tuple[str, str]) -> str:
    return f"{unit[0]} / {unit[1]}"


def compact_unit_label(unit: tuple[str, str]) -> str:
    return f"{unit[0]}\n{unit[1]}"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def primary_rows() -> dict[tuple[str, str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in read_tsv(SUMMARY_PATH):
        if row["contrast_role"] != "primary" or row["metric"] not in METRICS:
            continue
        key = (row["dataset"], row["model"], row["metric"], row["contrast_id"])
        result[key] = row
    if len(result) != 36:
        raise RuntimeError(f"Expected 36 primary result rows, found {len(result)}")
    return result


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_effect_forest(rows: dict[tuple[str, str, str, str], dict[str, str]]) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), sharey=True)
    y = np.arange(len(UNITS))
    styles = {
        "correct_text_vs_random": ("#2563EB", "o", -0.13),
        "correct_text_vs_shuffled_text": ("#D97706", "D", 0.13),
    }

    for ax, metric in zip(axes, METRICS):
        for contrast in CONTRASTS:
            color, marker, offset = styles[contrast]
            for index, (dataset, model) in enumerate(UNITS):
                row = rows[(dataset, model, metric, contrast)]
                mean = float(row["mean_difference"])
                lower = float(row["unadjusted_95_percent_ci_lower"])
                upper = float(row["unadjusted_95_percent_ci_upper"])
                significant = row["holm_reject_familywise_0_05"] == "true"
                ax.errorbar(
                    mean,
                    index + offset,
                    xerr=np.array([[mean - lower], [upper - mean]]),
                    fmt=marker,
                    color=color,
                    markerfacecolor=color if significant else "white",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    markersize=7,
                    capsize=3,
                    linewidth=1.3,
                    zorder=3,
                )
        ax.axvline(0, color="#111827", linewidth=1.0, zorder=2)
        ax.set_title(METRIC_TITLES[metric], fontweight="bold")
        ax.set_xlabel("Paired mean effect (positive = correct text better)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y, [unit_label(unit) for unit in UNITS])
    axes[0].invert_yaxis()
    legend = [
        Line2D([0], [0], marker="o", color="#2563EB", markerfacecolor="#2563EB", linewidth=0, label="Correct text vs random"),
        Line2D([0], [0], marker="D", color="#D97706", markerfacecolor="#D97706", linewidth=0, label="Correct text vs shuffled text"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="#374151", linewidth=0, label="Filled: Holm significant"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="white", linewidth=0, label="Open: not significant"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Effect of correct-text initialization across datasets and models", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.94,
        "Five paired seeds per comparison; error bars show unadjusted 95% paired t confidence intervals.",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.87, wspace=0.22)
    return list(save_figure(fig, "Figure_1_Primary_Effect_Forest"))


def condition_summary() -> dict[tuple[str, str, str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in read_tsv(RUNS_PATH):
        for metric in METRICS:
            grouped[(row["dataset"], row["model"], row["condition"], metric)].append(float(row[metric]))

    result: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    t_critical = 2.7764451051977987
    for key, values in grouped.items():
        if len(values) != 5:
            raise RuntimeError(f"Expected five seeds for {key}, found {len(values)}")
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        half_width = t_critical * sd / math.sqrt(len(values))
        result[key] = (mean, half_width)
    return result


def make_condition_means(summary: dict[tuple[str, str, str, str], tuple[float, float]]) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True)
    x = np.arange(len(UNITS))
    width = 0.23
    offsets = [-width, 0, width]

    for ax, metric in zip(axes, METRICS):
        for condition, offset in zip(CONDITIONS, offsets):
            means = [summary[(dataset, model, condition, metric)][0] for dataset, model in UNITS]
            errors = [summary[(dataset, model, condition, metric)][1] for dataset, model in UNITS]
            ax.bar(
                x + offset,
                means,
                width,
                yerr=errors,
                capsize=3,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                alpha=0.92,
                edgecolor="white",
                linewidth=0.5,
            )
        direction = "higher is better" if metric == "combined_filtered_test_mrr" else "lower is better"
        ax.set_title(f"{METRIC_TITLES[metric]} ({direction})", loc="left", fontweight="bold")
        ax.set_ylabel("Mean test value")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xticks(x, [compact_unit_label(unit) for unit in UNITS])
    axes[-1].tick_params(axis="x", pad=8)
    handles = [Patch(facecolor=CONDITION_COLORS[c], label=CONDITION_LABELS[c]) for c in CONDITIONS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Official test performance by initialization condition", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.955,
        "Bars show means over five seeds; error bars show 95% t confidence intervals across seeds.",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.91, hspace=0.28)
    return list(save_figure(fig, "Figure_2_Condition_Means"))


def format_effect(metric: str, value: float, significant: bool) -> str:
    decimals = 4 if metric == "combined_filtered_test_mrr" else 3
    suffix = "*" if significant else ""
    return f"{value:+.{decimals}f}{suffix}"


def make_significance_matrix(rows: dict[tuple[str, str, str, str], dict[str, str]]) -> list[Path]:
    column_keys: list[tuple[str, str]] = []
    column_labels: list[str] = []
    short_metric = {
        "combined_filtered_test_mrr": "MRR",
        "calibrated_test_multiclass_nll": "NLL",
        "calibrated_test_eaurc": "E-AURC",
    }
    for metric in METRICS:
        for contrast in CONTRASTS:
            column_keys.append((metric, contrast))
            control = "random" if contrast.endswith("random") else "shuffled"
            column_labels.append(f"{short_metric[metric]}\nvs {control}")

    states = np.zeros((len(UNITS), len(column_keys)), dtype=int)
    annotations: list[list[str]] = []
    for row_index, (dataset, model) in enumerate(UNITS):
        annotation_row: list[str] = []
        for column_index, (metric, contrast) in enumerate(column_keys):
            row = rows[(dataset, model, metric, contrast)]
            value = float(row["mean_difference"])
            significant = row["holm_reject_familywise_0_05"] == "true"
            states[row_index, column_index] = 1 if significant and value > 0 else (-1 if significant and value < 0 else 0)
            annotation_row.append(format_effect(metric, value, significant))
        annotations.append(annotation_row)

    cmap = ListedColormap(["#FCA5A5", "#E5E7EB", "#86EFAC"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    ax.imshow(states, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(UNITS)), [unit_label(unit) for unit in UNITS])
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for row_index in range(len(UNITS)):
        for column_index in range(len(column_keys)):
            ax.text(column_index, row_index, annotations[row_index][column_index], ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(UNITS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)
    legend = [
        Patch(facecolor="#86EFAC", label="Significant benefit"),
        Patch(facecolor="#FCA5A5", label="Significant harm"),
        Patch(facecolor="#E5E7EB", label="Not significant"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Direction and significance of correct-text effects", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Cell values are paired mean effects oriented so positive values favor correct text; * denotes Holm-adjusted significance.",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.14, top=0.83)
    return list(save_figure(fig, "Figure_3_Significance_Map"))


def write_guide() -> Path:
    path = OUT_DIR / "Figure_Captions_and_Interpretation.md"
    path.write_text(
        """# Official-result figures: captions and interpretation

## Figure 1 — Primary effect forest plot

**Suggested caption.** Paired effects of correct-text entity initialization relative to random and shuffled-text controls across three datasets and two knowledge-graph embedding models. Points show the mean paired difference over five official seeds and error bars show unadjusted 95% paired *t* confidence intervals. All differences are oriented so that positive values favor correct-text initialization. Filled markers indicate rejection after Holm correction within the corresponding confirmatory family; open markers indicate non-significant comparisons.

**Use in the thesis.** This should be the main Results figure because it shows effect direction, magnitude, uncertainty, both controls, and multiplicity-corrected significance in one place.

## Figure 2 — Condition means

**Suggested caption.** Official test-set performance under random, correct-text, and shuffled-text entity initialization. Bars show means over five seeds and error bars show 95% *t* confidence intervals across seeds. Higher MRR is better, whereas lower calibrated NLL and lower calibrated E-AURC are better.

**Use in the thesis.** Use this as a descriptive companion figure. It helps readers understand the absolute scale of each metric; inferential claims should still be based on the paired tests in Figure 1.

## Figure 3 — Significance map

**Suggested caption.** Direction and Holm-adjusted significance of the correct-text effects. Green cells indicate a significant benefit, red cells a significant harm, and grey cells no statistically significant difference. Values are paired mean effects oriented so that positive values favor correct-text initialization; an asterisk denotes Holm-adjusted significance.

**Use in the thesis.** This is an effective overview for the Discussion or presentation. It immediately shows that the effects are heterogeneous rather than uniformly beneficial.

## Interpretation summary

- **Link-prediction accuracy:** Correct text is not uniformly beneficial. It improves DistMult on CoDEx-M, has a very large negative effect on DistMult on WN18RR, and produces smaller mixed effects elsewhere.
- **Calibrated NLL:** TransE improves significantly in all six confirmatory comparisons, but DistMult is mixed and is significantly worse on FB15k-237 and CoDEx-M.
- **Calibrated E-AURC:** No confirmatory comparison shows a significant improvement. Eight comparisons show a significant deterioration and four are non-significant.
- **Central conclusion:** Text semantics affect the learned representation, but semantic correctness alone does not reliably improve uncertainty quality. The effect depends on the dataset, KGE architecture, uncertainty metric, and control condition.

## Reporting caution

Do not describe the study as showing a general uncertainty improvement. The strongest defensible claim is that correct-text initialization produces systematic but heterogeneous effects, including a disconnect between probability calibration (NLL) and selective-ranking quality (E-AURC).
""",
        encoding="utf-8",
    )
    return path


def package(files: list[Path]) -> Path:
    zip_path = OUT_DIR / "Thesis_Official_Result_Figures.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return zip_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = primary_rows()
    files: list[Path] = []
    files.extend(make_effect_forest(rows))
    files.extend(make_condition_means(condition_summary()))
    files.extend(make_significance_matrix(rows))
    files.append(write_guide())
    zip_path = package(files)
    for path in files + [zip_path]:
        print(path)


if __name__ == "__main__":
    main()
