import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                       'artifacts', 'revision', 'figures')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.9, 3.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis('off')


def box(x, y, w, h, title, items, title_size=11, item_size=10):
    bbox = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.3,rounding_size=0.6",
                          linewidth=1.0, edgecolor='black',
                          facecolor='white')
    ax.add_patch(bbox)
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top',
            fontsize=title_size, fontweight='bold')
    for i, item in enumerate(items):
        ax.text(x + w / 2, y + h - 4.2 - i * 2.2, item, ha='center',
                va='top', fontsize=item_size)


def arrow(x1, y1, x2, y2):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->,head_length=4,head_width=3',
                        linewidth=1.0, color='black',
                        mutation_scale=1)
    ax.add_patch(ar)


# Column 1: Models
box(0.5, 14, 19, 19, "5 model families",
    ["Gemma-2-9B-IT", "Llama-3.1-8B-Instruct",
     "Qwen3-4B-Instruct", "Phi-4 (14B)", "Qwen2.5-14B-Instruct"])

# Column 2: Adaptations
box(22, 14, 18, 19, "5 configurations",
    ["Base FP16", "Base 4-bit (NF4)", "LoRA (r = 16)",
     "QLoRA (4-bit + LoRA)", "FFT"])
ax.text(31, 11.8, "Adapters/FFT: 1 epoch on",
        ha='center', va='top', fontsize=9.5, style='italic')
ax.text(31, 9.6, "Alpaca-cleaned (51,760)",
        ha='center', va='top', fontsize=9.5, style='italic')

# Column 3: Attacks (grouped)
box(42.5, 14, 22, 19, "Attacks",
    ["Black-box: PAIR,", "DeepInception, ArtPrompt",
     "White-box: AutoDAN,",
     "GCG (5-family JBB probe)",
     "Multi-turn:", "ActorAttack"])
ax.text(53.5, 11.8, "HarmBench-400 + JBB-100",
        ha='center', va='top', fontsize=9.5, style='italic')
ax.text(53.5, 9.6, "(GCG: JBB; ActorAttack: HB)",
        ha='center', va='top', fontsize=9.5, style='italic')

# Column 4: Evaluation
box(66.5, 18, 19, 15, "Evaluation",
    ["HB-Cls", "GPT-4o-mini", "LlamaGuard-3"])

# Human annotation (separate box below Evaluation, no overlap)
box(66.5, 1, 19, 13, "Human annotation",
    ["N = 750", "(500 ASR + 250 ORR)",
     "two annotators",
     "kappa = 0.62 / 0.89"],
    title_size=10.5, item_size=10)

# Outputs box (right of Evaluation; tall enough for all 8 text lines)
box(87.5, 10, 12, 23, "Outputs",
    ["ASR", "(HarmBench", "+ JBB)", "", "ORR", "(OR-Bench", "Hard-1K /", "Toxic)"],
    title_size=11, item_size=10)

# Arrows between columns
arrow(19.7, 23.5, 21.8, 23.5)
arrow(40.2, 23.5, 42.3, 23.5)
arrow(64.7, 23.5, 66.3, 23.5)
arrow(85.7, 23.5, 87.3, 23.5)

# Calibration arrow: Human annotation -> Evaluation
arrow(73.5, 14.2, 73.5, 17.8)
ax.text(74.7, 16, "validates", ha='left', va='center',
        fontsize=9.5, style='italic')

plt.tight_layout()
pdf_path = os.path.join(OUT_DIR, 'figure1_pipeline.pdf')
png_path = os.path.join(OUT_DIR, 'figure1_pipeline.png')
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)
plt.savefig(png_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
print(f"Saved {pdf_path} and {png_path}")
