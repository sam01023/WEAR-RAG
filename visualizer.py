"""
WEAR-RAG — Visualizer (v2)
===========================
Charts:
    1. Evidence Importance bar chart (ASCII + matplotlib)
    2. Score Component Breakdown table
    3. Pipeline Comparison grouped bar (EM, F1, ROUGE-L, BLEU, MRR, RetPrec)
    4. Per-Metric breakdown chart (one subplot per metric)
    5. Score Distribution histogram
"""

import logging
from typing import Dict, List, Optional

from evidence_aggregator import EvidenceItem
from evaluator import EvaluationReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ASCII Visualizer
# ---------------------------------------------------------------------------

class ASCIIVisualizer:
    BAR_WIDTH = 40
    FILL = "█"
    EMPTY = "░"

    def evidence_chart(self, items: List[EvidenceItem], title: str = "Evidence Importance") -> str:
        if not items:
            return "No evidence to display."
        max_score = max(item.evidence_score for item in items) or 1.0
        lines = ["", f"  {title}", "  " + "─" * (self.BAR_WIDTH + 30)]
        for item in items:
            fill = int((item.evidence_score / max_score) * self.BAR_WIDTH)
            bar = self.FILL * fill + self.EMPTY * (self.BAR_WIDTH - fill)
            label = f"[{item.source_doc_id[:15]:<15}]"
            lines.append(f"  {label} {bar} {item.evidence_score:.3f}")
        lines.append("  " + "─" * (self.BAR_WIDTH + 30) + "\n")
        return "\n".join(lines)

    def score_breakdown(self, items: List[EvidenceItem]) -> str:
        if not items:
            return "No evidence."
        header = f"\n  {'Rank':<5} {'Total':<8} {'Sim':<8} {'Reranker':<10} {'Density':<10} Source"
        sep = "  " + "─" * 65
        rows = [header, sep]
        for item in items:
            rows.append(
                f"  {item.evidence_rank:<5} {item.evidence_score:<8.3f} "
                f"{item.similarity_score:<8.3f} {item.reranker_score:<10.3f} "
                f"{item.density_score:<10.3f} [{item.source_doc_id[:20]}]"
            )
        rows.append(sep)
        return "\n".join(rows)

    def comparison_table(self, reports: List[EvaluationReport]) -> str:
        header = f"\n  {'System':<22} {'EM':>7} {'F1':>7} {'ROUGE-L':>9} {'BLEU-1':>8} {'MRR':>7} {'RetPrec':>9}"
        sep = "  " + "─" * 72
        rows = [header, sep]
        for r in reports:
            rows.append(
                f"  {r.system_name:<22} "
                f"{r.avg_exact_match:>7.4f} {r.avg_f1:>7.4f} "
                f"{r.avg_rouge_l:>9.4f} {r.avg_bleu_1:>8.4f} "
                f"{r.avg_mrr:>7.4f} {r.avg_retrieval_precision:>9.4f}"
            )
        rows.append(sep)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Matplotlib Visualizer
# ---------------------------------------------------------------------------

class MatplotlibVisualizer:

    # Extended palette for up to 10 systems
    COLORS = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    ]

    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize=(14, 5)):
        self.figsize = figsize
        self._ascii = ASCIIVisualizer()
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            try:
                plt.style.use(style)
            except Exception:
                plt.style.use("ggplot")
            self.plt = plt
            self._available = True
        except ImportError:
            self._available = False

    def evidence_importance(self, items, title="Evidence Importance (WEAR-RAG)", save_path=None):
        if not self._available or not items:
            print(self._ascii.evidence_chart(items, title))
            return None
        plt = self.plt
        import numpy as np
        n = len(items)
        y = np.arange(n)
        sims   = [item.similarity_score * item.weights.get("similarity", 0.5) for item in items]
        ranks  = [item.reranker_score   * item.weights.get("reranker",   0.4) for item in items]
        dens   = [item.density_score    * item.weights.get("density",    0.1) for item in items]
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.barh(y, sims, align="center", label="Similarity (×0.5)", color=self.COLORS[0], height=0.5)
        ax.barh(y, ranks, left=sims, align="center", label="Reranker (×0.4)", color=self.COLORS[1], height=0.5)
        ax.barh(y, dens, left=[s+r for s,r in zip(sims,ranks)], align="center", label="Density (×0.1)", color=self.COLORS[2], height=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"Rank {item.evidence_rank}" for item in items], fontsize=9)
        ax.set_xlabel("Evidence Score", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved: %s", save_path)
        return fig

    def pipeline_comparison(self, reports, title="WEAR-RAG vs Baselines", save_path=None):
        if not self._available:
            print(self._ascii.comparison_table(reports))
            return None
        plt = self.plt
        import numpy as np

        metrics = ["Exact Match", "F1 Score", "ROUGE-L", "BLEU-1", "MRR", "Ret. Precision"]
        systems = [r.system_name for r in reports]
        values = [
            [r.avg_exact_match, r.avg_f1, r.avg_rouge_l,
             r.avg_bleu_1, r.avg_mrr, r.avg_retrieval_precision]
            for r in reports
        ]

        n_systems = len(systems)
        n_metrics = len(metrics)
        x = np.arange(n_metrics)
        width = max(0.08, 0.8 / n_systems)  # dynamic bar width

        fig_width = max(14, n_metrics * 2 + n_systems)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        for i, (system, vals) in enumerate(zip(systems, values)):
            color = self.COLORS[i % len(self.COLORS)]
            offset = (i - n_systems / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=system, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, ncol=min(4, n_systems), loc="upper right")
        ax.yaxis.grid(True, alpha=0.4)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved: %s", save_path)
        return fig

    def metric_breakdown(self, reports, title="All Metrics — System Comparison", save_path=None):
        """One subplot per metric showing all systems side by side."""
        if not self._available:
            return None
        plt = self.plt
        import numpy as np

        metric_labels = ["EM", "F1", "ROUGE-L", "BLEU-1", "MRR", "Ret. Precision"]
        metric_attrs  = ["avg_exact_match", "avg_f1", "avg_rouge_l",
                         "avg_bleu_1", "avg_mrr", "avg_retrieval_precision"]
        systems = [r.system_name for r in reports]
        n_systems = len(systems)

        fig_width = max(18, len(metric_labels) * 3)
        fig, axes = plt.subplots(1, len(metric_labels), figsize=(fig_width, 5), sharey=False)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for ax, label, attr in zip(axes, metric_labels, metric_attrs):
            vals = [getattr(r, attr) for r in reports]
            colors = [self.COLORS[i % len(self.COLORS)] for i in range(n_systems)]
            bars = ax.bar(range(n_systems), vals, color=colors, alpha=0.85, width=0.6)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xticks(range(n_systems))
            ax.set_xticklabels([s.replace(" ", "\n") for s in systems], fontsize=5, rotation=45, ha="right")
            ax.set_ylim(0, max(max(vals) * 1.3, 0.1))
            ax.yaxis.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved: %s", save_path)
        return fig

    def score_distribution(self, all_scores, threshold=0.3, title="Evidence Score Distribution", save_path=None):
        if not self._available or not all_scores:
            return None
        plt = self.plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(all_scores, bins=20, color="#4C72B0", alpha=0.75, edgecolor="white")
        ax.axvline(threshold, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
        ax.set_xlabel("Evidence Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved: %s", save_path)
        return fig
