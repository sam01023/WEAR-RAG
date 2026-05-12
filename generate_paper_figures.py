"""
Generate all figures for the WEAR-RAG IEEE research paper.
Outputs publication-quality PNG images for LaTeX inclusion.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(OUT, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────
C_BASELINE = "#5B9BD5"
C_WEARRAG  = "#ED7D31"
C_DARK     = "#2E2E2E"
C_GRID     = "#E0E0E0"
C_BG       = "#FFFFFF"

FONT = {'family': 'serif', 'size': 10}
matplotlib.rc('font', **FONT)
matplotlib.rc('axes', linewidth=0.8)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('ytick', direction='in')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Pipeline Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════

def fig_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(7.16, 3.5))  # IEEE column width
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    stages = [
        (0.3, 2.5, "Documents\n(Corpus)", "#F2F2F2", "#666"),
        (1.8, 2.5, "Semantic\nChunking", "#D6EAF8", "#2874A6"),
        (3.3, 2.5, "Query\nDecomposition", "#D5F5E3", "#1E8449"),
        (4.8, 2.5, "Dense\nRetrieval\n(FAISS)", "#FCF3CF", "#B7950B"),
        (6.3, 2.5, "Cross-Encoder\nReranking", "#FADBD8", "#C0392B"),
        (7.8, 2.5, "Weighted\nEvidence\nAggregation", "#E8DAEF", "#7D3C98"),
        (9.3, 2.5, "Grounded\nGeneration", "#FDEBD0", "#CA6F1E"),
    ]

    box_w, box_h = 1.2, 1.6

    for i, (x, y, label, bg, border) in enumerate(stages):
        rect = mpatches.FancyBboxPatch(
            (x - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=7,
                fontweight='bold', color=C_DARK, linespacing=1.3)

        # Stage number badge
        badge = plt.Circle((x - box_w/2 + 0.12, y + box_h/2 - 0.12), 0.12,
                           color=border, zorder=5)
        ax.add_patch(badge)
        ax.text(x - box_w/2 + 0.12, y + box_h/2 - 0.12, str(i),
                ha='center', va='center', fontsize=6, color='white', fontweight='bold', zorder=6)

        # Arrow to next stage
        if i < len(stages) - 1:
            nx = stages[i+1][0]
            ax.annotate('', xy=(nx - box_w/2 - 0.05, y), xytext=(x + box_w/2 + 0.05, y),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    # Input/output labels
    ax.text(0.3, 0.6, "Input: Question q + Corpus D", ha='center', va='center',
            fontsize=7.5, fontstyle='italic', color='#555')
    ax.text(9.3, 0.6, "Output: Answer a + Evidence E", ha='center', va='center',
            fontsize=7.5, fontstyle='italic', color='#555')

    # Title
    ax.text(5.0, 4.6, "WEAR-RAG Pipeline Architecture", ha='center', va='center',
            fontsize=11, fontweight='bold', color=C_DARK)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.1)
    plt.close()
    print("  ✓ fig_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Main Performance Comparison (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def fig_performance_comparison():
    metrics = ['Exact\nMatch', 'F1', 'ROUGE-L', 'BLEU-1', 'Retrieval\nPrecision']
    baseline = [0.0700, 0.2136, 0.2123, 0.1582, 0.3480]
    wearrag  = [0.0800, 0.2339, 0.2330, 0.1733, 0.3960]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.16, 3.8))

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline RAG',
                   color=C_BASELINE, edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, wearrag, width, label='WEAR-RAG',
                   color=C_WEARRAG, edgecolor='white', linewidth=0.5, zorder=3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, color=C_DARK)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, color=C_DARK, fontweight='bold')

    ax.set_ylabel('Score', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 0.50)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color=C_GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Performance Comparison: Baseline RAG vs. WEAR-RAG (n=100)',
                 fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_performance.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.1)
    plt.close()
    print("  ✓ fig_performance.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Relative Improvement (%)
# ═══════════════════════════════════════════════════════════════════════════

def fig_relative_improvement():
    metrics = ['EM', 'F1', 'ROUGE-L', 'BLEU-1', 'MRR', 'Ret. Precision']
    deltas  = [14.29, 9.50, 9.75, 9.54, -0.08, 13.79]
    colors  = [C_WEARRAG if d > 0 else C_BASELINE for d in deltas]

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    y_pos = np.arange(len(metrics))

    bars = ax.barh(y_pos, deltas, color=colors, edgecolor='white', linewidth=0.5, height=0.55, zorder=3)

    for bar, d in zip(bars, deltas):
        w = bar.get_width()
        offset = 0.4 if d >= 0 else -0.4
        ha = 'left' if d >= 0 else 'right'
        ax.text(w + offset, bar.get_y() + bar.get_height()/2,
                f'{d:+.2f}%', ha=ha, va='center', fontsize=8.5, fontweight='bold', color=C_DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel('Relative Improvement (%)', fontsize=10)
    ax.axvline(x=0, color='#888', linewidth=0.8, linestyle='-')
    ax.grid(axis='x', linestyle='--', alpha=0.3, color=C_GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Relative Improvement of WEAR-RAG over Baseline RAG',
                 fontsize=11, fontweight='bold', pad=10)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_relative.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.1)
    plt.close()
    print("  ✓ fig_relative.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Evidence Score Composition (stacked bar — showing weight
#           contribution of sim, rank, density)
# ═══════════════════════════════════════════════════════════════════════════

def fig_evidence_composition():
    chunks = ['Chunk A\n(Relevant)', 'Chunk B\n(Relevant)', 'Chunk C\n(Marginal)',
              'Chunk D\n(Noise)', 'Chunk E\n(Noise)']

    # Simulated component scores × weights
    sim_contrib   = [0.42, 0.38, 0.25, 0.15, 0.10]  # w=0.5
    rank_contrib  = [0.36, 0.30, 0.12, 0.06, 0.04]  # w=0.4
    dens_contrib  = [0.08, 0.07, 0.05, 0.03, 0.02]  # w=0.1

    x = np.arange(len(chunks))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7.16, 3.5))

    b1 = ax.bar(x, sim_contrib, width, label='Similarity (w=0.5)',
                color='#5DADE2', edgecolor='white', linewidth=0.5, zorder=3)
    b2 = ax.bar(x, rank_contrib, width, bottom=sim_contrib,
                label='Reranker (w=0.4)',
                color='#F39C12', edgecolor='white', linewidth=0.5, zorder=3)
    b3 = ax.bar(x, dens_contrib, width,
                bottom=[s+r for s,r in zip(sim_contrib, rank_contrib)],
                label='Density (w=0.1)',
                color='#58D68D', edgecolor='white', linewidth=0.5, zorder=3)

    # Threshold line
    ax.axhline(y=0.3, color='#E74C3C', linewidth=1.5, linestyle='--', zorder=4)
    ax.text(4.55, 0.31, 'θ = 0.3', fontsize=8, color='#E74C3C', fontweight='bold')

    # Total score labels
    totals = [s+r+d for s,r,d in zip(sim_contrib, rank_contrib, dens_contrib)]
    for i, t in enumerate(totals):
        ax.text(i, t + 0.01, f'{t:.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=C_DARK)

    ax.set_ylabel('Evidence Score S(c)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(chunks, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=C_GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Weighted Evidence Score Composition and Threshold Filtering',
                 fontsize=11, fontweight='bold', pad=12)

    # Shade filtered region
    ax.axhspan(0, 0.3, alpha=0.05, color='red', zorder=1)
    ax.text(0.0, 0.14, 'FILTERED', fontsize=9, color='#E74C3C', alpha=0.6,
            fontstyle='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_evidence_scores.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.1)
    plt.close()
    print("  ✓ fig_evidence_scores.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Semantic Chunking vs Fixed Chunking (conceptual)
# ═══════════════════════════════════════════════════════════════════════════

def fig_chunking_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # --- Fixed Chunking (left) ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('(a) Fixed-Length Chunking', fontsize=9, fontweight='bold', pad=8)

    colors_fixed = ['#D6EAF8', '#D5F5E3', '#FCF3CF', '#FADBD8']
    topics = ['Topic A', '← split mid-sentence →', 'Topic B mixed', 'Topic B cont.']
    for i, (c, t) in enumerate(zip(colors_fixed, topics)):
        y = 3.2 - i * 0.85
        rect = mpatches.FancyBboxPatch((0.5, y - 0.3), 9, 0.6,
               boxstyle="round,pad=0.05", facecolor=c, edgecolor='#999', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(5, y, f'Chunk {i+1}: {t}', ha='center', va='center', fontsize=7.5)

    # Cut mark
    ax.plot([5, 5], [2.55, 2.9], 'r--', linewidth=1.5)
    ax.text(5.8, 2.72, '✗ arbitrary\n   break', fontsize=6.5, color='red')

    # --- Semantic Chunking (right) ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('(b) Semantic Chunking (WEAR-RAG)', fontsize=9, fontweight='bold', pad=8)

    colors_sem = ['#D6EAF8', '#D5F5E3', '#FCF3CF']
    topics_sem = ['Topic A (complete)', 'Topic B (complete)', 'Topic C (complete)']
    heights = [0.75, 0.75, 0.55]
    y_start = 3.3
    for i, (c, t, h) in enumerate(zip(colors_sem, topics_sem, heights)):
        y = y_start - sum(heights[:i]) - i*0.15 - h/2
        rect = mpatches.FancyBboxPatch((0.5, y - h/2), 9, h,
               boxstyle="round,pad=0.05", facecolor=c, edgecolor='#2E86C1', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(5, y, f'Chunk {i+1}: {t}', ha='center', va='center', fontsize=7.5)

    ax.text(5, 0.3, '✓ topic-coherent boundaries', fontsize=7, color='#1E8449',
            ha='center', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_chunking.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.1)
    plt.close()
    print("  ✓ fig_chunking.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: MRR vs Retrieval Precision scatter/bar comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_mrr_vs_precision():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # MRR comparison
    systems = ['Baseline\nRAG', 'WEAR-\nRAG']
    mrr_vals = [0.9558, 0.9550]
    ax1.bar(systems, mrr_vals, color=[C_BASELINE, C_WEARRAG], width=0.5,
            edgecolor='white', zorder=3)
    ax1.set_ylim(0.90, 1.0)
    ax1.set_ylabel('MRR', fontsize=10)
    ax1.set_title('(a) Mean Reciprocal Rank', fontsize=9, fontweight='bold')
    for i, v in enumerate(mrr_vals):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_axisbelow(True)

    # Retrieval precision
    rp_vals = [0.3480, 0.3960]
    ax2.bar(systems, rp_vals, color=[C_BASELINE, C_WEARRAG], width=0.5,
            edgecolor='white', zorder=3)
    ax2.set_ylim(0, 0.5)
    ax2.set_ylabel('Retrieval Precision', fontsize=10)
    ax2.set_title('(b) Retrieval Precision', fontsize=9, fontweight='bold')
    for i, v in enumerate(rp_vals):
        ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_axisbelow(True)

    # Overall title
    fig.suptitle('Retrieval Quality: Similar First-Hit, Better Overall Evidence',
                 fontsize=10.5, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_retrieval.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.1)
    plt.close()
    print("  ✓ fig_retrieval.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Radar chart — multi-metric comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_radar_chart():
    categories = ['EM', 'F1', 'ROUGE-L', 'BLEU-1', 'MRR', 'Ret.\nPrec.']
    N = len(categories)

    # Normalize all to [0,1] for radar (MRR is already near 1, others scaled)
    baseline_raw = [0.0700, 0.2136, 0.2123, 0.1582, 0.9558, 0.3480]
    wearrag_raw  = [0.0800, 0.2339, 0.2330, 0.1733, 0.9550, 0.3960]

    # Scale for visibility (MRR is already near 1; others are small)
    # Use raw values but set radar max to 1.0
    baseline = baseline_raw + [baseline_raw[0]]
    wearrag  = wearrag_raw + [wearrag_raw[0]]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))

    ax.plot(angles, baseline, 'o-', linewidth=1.5, color=C_BASELINE, label='Baseline RAG', markersize=4)
    ax.fill(angles, baseline, alpha=0.15, color=C_BASELINE)
    ax.plot(angles, wearrag, 's-', linewidth=1.5, color=C_WEARRAG, label='WEAR-RAG', markersize=4)
    ax.fill(angles, wearrag, alpha=0.15, color=C_WEARRAG)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, color='#888')
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=9, framealpha=0.9)
    ax.set_title('Multi-Metric Performance Radar', fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_radar.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.15)
    plt.close()
    print("  ✓ fig_radar.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Module dependency / architecture block diagram
# ═══════════════════════════════════════════════════════════════════════════

def fig_module_architecture():
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def draw_box(x, y, w, h, label, color, border):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06", facecolor=color, edgecolor=border, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color=C_DARK, linespacing=1.2)

    # Core pipeline modules (middle row)
    draw_box(0.3, 3.2, 1.8, 0.9, 'Document\nProcessor', '#D6EAF8', '#2874A6')
    draw_box(2.4, 3.2, 1.6, 0.9, 'Embedding\nEngine', '#D6EAF8', '#2874A6')
    draw_box(4.3, 3.2, 1.7, 0.9, 'Vector\nStore', '#D6EAF8', '#2874A6')
    draw_box(6.3, 3.2, 1.6, 0.9, 'Cross-Encoder\nReranker', '#D6EAF8', '#2874A6')
    draw_box(8.2, 3.2, 1.8, 0.9, 'Evidence\nAggregator', '#E8DAEF', '#7D3C98')
    draw_box(10.3, 3.2, 1.4, 0.9, 'LLM\nGenerator', '#D6EAF8', '#2874A6')

    # Top row: orchestration
    draw_box(3.5, 5.5, 2.0, 0.9, 'Pipeline\nOrchestrator', '#FDEBD0', '#CA6F1E')
    draw_box(6.0, 5.5, 2.0, 0.9, 'Configuration\nManager', '#FCF3CF', '#B7950B')

    # Bottom row: support modules
    draw_box(0.5, 0.8, 2.0, 0.9, 'Query\nDecomposer', '#D5F5E3', '#1E8449')
    draw_box(3.0, 0.8, 1.8, 0.9, 'Evaluator\n(Metrics)', '#FADBD8', '#C0392B')
    draw_box(5.2, 0.8, 1.8, 0.9, 'Visualizer\n(Charts)', '#FADBD8', '#C0392B')
    draw_box(7.4, 0.8, 1.5, 0.9, 'Test Suite', '#FADBD8', '#C0392B')
    draw_box(9.3, 0.8, 1.5, 0.9, 'Web API\n(Flask)', '#FDEBD0', '#CA6F1E')

    # Arrows from orchestrator down to pipeline
    for tx in [1.2, 3.2, 5.15, 7.1, 9.1, 11.0]:
        ax.annotate('', xy=(tx, 4.1), xytext=(4.5, 5.5),
                    arrowprops=dict(arrowstyle='->', color='#AAA', lw=0.6, connectionstyle='arc3,rad=0.0'))

    # Layer labels
    ax.text(0.1, 6.6, 'Orchestration Layer', fontsize=8, fontstyle='italic', color='#888')
    ax.text(0.1, 4.3, 'Core Pipeline Layer', fontsize=8, fontstyle='italic', color='#888')
    ax.text(0.1, 1.9, 'Support Layer', fontsize=8, fontstyle='italic', color='#888')

    # Title
    ax.set_title('WEAR-RAG Component Architecture', fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_modules.png"), dpi=300, bbox_inches='tight',
                facecolor=C_BG, pad_inches=0.1)
    plt.close()
    print("  ✓ fig_modules.png")


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating WEAR-RAG paper figures...")
    fig_pipeline_architecture()
    fig_performance_comparison()
    fig_relative_improvement()
    fig_evidence_composition()
    fig_chunking_comparison()
    fig_mrr_vs_precision()
    fig_radar_chart()
    fig_module_architecture()
    print(f"\nAll figures saved to: {OUT}")
