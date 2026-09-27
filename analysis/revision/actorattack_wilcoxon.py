"""Wilcoxon signed-rank test for ActorAttack base-vs-fine-tuned cells.

The paired inputs are the HB-Cls base-vs-fine-tuned averages for all five
families, as reported in the ActorAttack scoreboard appendix. Base is the mean
of base_fp16 and base_4bit. Ten pairs: five families x {LoRA, FFT}.
"""

from pathlib import Path

from scipy.stats import wilcoxon


PAIRS = [
    ("Gemma-2 LoRA", 12.72, 16.79),
    ("Gemma-2 FFT", 12.72, 17.81),
    ("Phi-4 LoRA", 6.36, 11.45),
    ("Phi-4 FFT", 6.36, 9.41),
    ("Llama-3.1 LoRA", 11.20, 12.72),
    ("Llama-3.1 FFT", 11.20, 13.23),
    ("Qwen-2.5 LoRA", 12.21, 11.20),
    ("Qwen-2.5 FFT", 12.21, 10.94),
    ("Qwen-3 LoRA", 4.83, 8.91),
    ("Qwen-3 FFT", 4.83, 8.65),
]


def main() -> None:
    base = [row[1] for row in PAIRS]
    fine_tuned = [row[2] for row in PAIRS]
    result = wilcoxon(base, fine_tuned)
    output = f"W={result.statistic:.1f}\np={result.pvalue:.6f}\n"
    out_path = Path("artifacts/revision/actorattack_wilcoxon.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
