"""Regenerate fig_before_after.png with all content inside the figure bounds."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = os.path.join(os.path.dirname(__file__), "paper_figures")
C_DARK = "#2E2E2E"

fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.5))

# ── Left panel: Baseline RAG ────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('(a) Baseline RAG Evidence', fontsize=9.5, fontweight='bold', pad=10)

# Query box
q_rect = mpatches.FancyBboxPatch((0.3, 8.2), 9.4, 1.2,
        boxstyle="round,pad=0.1", facecolor='#D6EAF8', edgecolor='#2874A6', linewidth=1.5)
ax.add_patch(q_rect)
ax.text(5, 8.8, 'Query: "Why are transformers\nbetter than RNNs?"',
        ha='center', va='center', fontsize=7.5, fontweight='bold', color='#2874A6')

# Chunk 1 - red
c1 = mpatches.FancyBboxPatch((0.3, 5.8), 9.4, 1.8,
        boxstyle="round,pad=0.08", facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=1.2)
ax.add_patch(c1)
ax.text(5, 7.0, 'Chunk 1: "RNNs were introduced in the\n1980s by Elman and Jordan..."',
        ha='center', va='center', fontsize=7, color=C_DARK)
ax.text(5, 6.15, '✗ Historical, not comparative',
        ha='center', va='center', fontsize=6.5, color='#E74C3C', fontstyle='italic')

# Chunk 2 - yellow
c2 = mpatches.FancyBboxPatch((0.3, 3.5), 9.4, 1.8,
        boxstyle="round,pad=0.08", facecolor='#FCF3CF', edgecolor='#F39C12', linewidth=1.2)
ax.add_patch(c2)
ax.text(5, 4.7, 'Chunk 2: "The BERT model uses 12\ntransformer encoder layers..."',
        ha='center', va='center', fontsize=7, color=C_DARK)
ax.text(5, 3.85, '✗ Partial — mentions transformers',
        ha='center', va='center', fontsize=6.5, color='#D4AC0D', fontstyle='italic')

# Chunk 3 - red
c3 = mpatches.FancyBboxPatch((0.3, 1.2), 9.4, 1.8,
        boxstyle="round,pad=0.08", facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=1.2)
ax.add_patch(c3)
ax.text(5, 2.4, 'Chunk 3: "Training GPT-3 required\n175 billion parameters..."',
        ha='center', va='center', fontsize=7, color=C_DARK)
ax.text(5, 1.55, '✗ Off-topic scaling info',
        ha='center', va='center', fontsize=6.5, color='#E74C3C', fontstyle='italic')

# Result label
ax.text(5, 0.4, 'Result: Vague, incomplete answer',
        ha='center', va='center', fontsize=8, fontweight='bold', color='#E74C3C')

# ── Right panel: WEAR-RAG ────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('(b) WEAR-RAG Curated Evidence', fontsize=9.5, fontweight='bold', pad=10)

# Query box
q_rect2 = mpatches.FancyBboxPatch((0.3, 8.2), 9.4, 1.2,
        boxstyle="round,pad=0.1", facecolor='#D6EAF8', edgecolor='#2874A6', linewidth=1.5)
ax.add_patch(q_rect2)
ax.text(5, 8.8, 'Query: "Why are transformers\nbetter than RNNs?"',
        ha='center', va='center', fontsize=7.5, fontweight='bold', color='#2874A6')

# Chunk 1 - green
c1g = mpatches.FancyBboxPatch((0.3, 5.8), 9.4, 1.8,
        boxstyle="round,pad=0.08", facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=1.2)
ax.add_patch(c1g)
ax.text(5, 7.0, 'Chunk 1: "Self-attention allows each\ntoken to attend to all others..."',
        ha='center', va='center', fontsize=7, color=C_DARK)
ax.text(5, 6.15, '✓ Self-attention  (S = 0.87)',
        ha='center', va='center', fontsize=6.5, color='#1E8449', fontweight='bold')

# Chunk 2 - green
c2g = mpatches.FancyBboxPatch((0.3, 3.5), 9.4, 1.8,
        boxstyle="round,pad=0.08", facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=1.2)
ax.add_patch(c2g)
ax.text(5, 4.7, 'Chunk 2: "Unlike RNNs, transformers\nprocess all tokens in parallel..."',
        ha='center', va='center', fontsize=7, color=C_DARK)
ax.text(5, 3.85, '✓ Parallelization  (S = 0.82)',
        ha='center', va='center', fontsize=6.5, color='#1E8449', fontweight='bold')

# Chunk 3 - green
c3g = mpatches.FancyBboxPatch((0.3, 1.2), 9.4, 1.8,
        boxstyle="round,pad=0.08", facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=1.2)
ax.add_patch(c3g)
ax.text(5, 2.4, 'Chunk 3: "Transformers capture long-\nrange dependencies via attention..."',
        ha='center', va='center', fontsize=7, color=C_DARK)
ax.text(5, 1.55, '✓ Long-range deps  (S = 0.79)',
        ha='center', va='center', fontsize=6.5, color='#1E8449', fontweight='bold')

# Result label
ax.text(5, 0.4, 'Result: Precise, grounded answer',
        ha='center', va='center', fontsize=8, fontweight='bold', color='#1E8449')

# ── Main title ──
fig.suptitle('Evidence Quality: Baseline RAG vs. WEAR-RAG',
             fontsize=12, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_before_after.png"), dpi=300, bbox_inches='tight',
            facecolor='white', pad_inches=0.15)
plt.close()
print("✓ fig_before_after.png regenerated")
