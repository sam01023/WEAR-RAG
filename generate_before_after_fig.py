"""Generate Before vs After evidence comparison figure for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(OUT, exist_ok=True)

def fig_before_after():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.8))

    # --- LEFT: Baseline RAG Evidence ---
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    ax1.set_title('(a) Baseline RAG Evidence', fontsize=9, fontweight='bold', pad=10)

    query_box = mpatches.FancyBboxPatch((0.3, 5.0), 9.4, 0.7,
        boxstyle="round,pad=0.08", facecolor='#EBF5FB', edgecolor='#2874A6', linewidth=1.5)
    ax1.add_patch(query_box)
    ax1.text(5, 5.35, 'Query: "Why are transformers better than RNNs?"',
             ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1B4F72')

    baseline_chunks = [
        ("Chunk 1: \"RNNs were introduced in the\n1980s by Elman and Jordan...\"", '#FADBD8', '#E74C3C', '✗ Historical, not comparative'),
        ("Chunk 2: \"The BERT model uses 12\ntransformer encoder layers...\"", '#FCF3CF', '#F39C12', '△ Partial — mentions transformers'),
        ("Chunk 3: \"Training GPT-3 required\n175 billion parameters...\"", '#FADBD8', '#E74C3C', '✗ Off-topic scaling info'),
    ]

    for i, (text, bg, border, verdict) in enumerate(baseline_chunks):
        y = 3.8 - i * 1.4
        rect = mpatches.FancyBboxPatch((0.3, y - 0.45), 6.0, 0.9,
            boxstyle="round,pad=0.05", facecolor=bg, edgecolor=border, linewidth=1.0)
        ax1.add_patch(rect)
        ax1.text(3.3, y, text, ha='center', va='center', fontsize=6.5, color='#2E2E2E', linespacing=1.2)
        ax1.text(8.5, y, verdict, ha='center', va='center', fontsize=6, color=border,
                 fontstyle='italic', fontweight='bold')

    ax1.text(5, 0.3, 'Result: Vague, incomplete answer', fontsize=7.5,
             ha='center', color='#C0392B', fontweight='bold', fontstyle='italic')

    # --- RIGHT: WEAR-RAG Evidence ---
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('(b) WEAR-RAG Curated Evidence', fontsize=9, fontweight='bold', pad=10)

    query_box2 = mpatches.FancyBboxPatch((0.3, 5.0), 9.4, 0.7,
        boxstyle="round,pad=0.08", facecolor='#EBF5FB', edgecolor='#2874A6', linewidth=1.5)
    ax2.add_patch(query_box2)
    ax2.text(5, 5.35, 'Query: "Why are transformers better than RNNs?"',
             ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1B4F72')

    wearrag_chunks = [
        ("Chunk 1: \"Self-attention allows each\ntoken to attend to all others...\"", '#D5F5E3', '#1E8449', '✓ Self-attention (S=0.87)'),
        ("Chunk 2: \"Unlike RNNs, transformers\nprocess all tokens in parallel...\"", '#D5F5E3', '#1E8449', '✓ Parallelization (S=0.82)'),
        ("Chunk 3: \"Transformers capture long-\nrange dependencies via attention...\"", '#D5F5E3', '#1E8449', '✓ Long-range deps (S=0.79)'),
    ]

    for i, (text, bg, border, verdict) in enumerate(wearrag_chunks):
        y = 3.8 - i * 1.4
        rect = mpatches.FancyBboxPatch((0.3, y - 0.45), 6.0, 0.9,
            boxstyle="round,pad=0.05", facecolor=bg, edgecolor=border, linewidth=1.0)
        ax2.add_patch(rect)
        ax2.text(3.3, y, text, ha='center', va='center', fontsize=6.5, color='#2E2E2E', linespacing=1.2)
        ax2.text(8.5, y, verdict, ha='center', va='center', fontsize=6, color=border,
                 fontstyle='italic', fontweight='bold')

    ax2.text(5, 0.3, 'Result: Precise, grounded answer', fontsize=7.5,
             ha='center', color='#1E8449', fontweight='bold', fontstyle='italic')

    fig.suptitle('Evidence Quality: Baseline RAG vs. WEAR-RAG',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_before_after.png"), dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()
    print("  ✓ fig_before_after.png")

if __name__ == "__main__":
    fig_before_after()
