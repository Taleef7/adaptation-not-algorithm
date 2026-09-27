#!/usr/bin/env python3
"""
ArtPrompt completions against a merged seed-42 (controlled-grid) checkpoint.

  * Test cases are reused from the original-grid cell of the same family and method.
    ArtPrompt masking is performed by gpt-4o-mini on the behavior text only
    (HarmBench baselines/artprompt/artprompt.py), so the target model is not involved in
    test-case generation and each seed-42 checkpoint is attacked with the same prompts as
    the corresponding original-grid cell.
  * Greedy decoding: do_sample=False, num_beams=1, max_new_tokens=512, matching the HarmBench
    HF generation path in generate_completions.py. ArtPrompt completions are deterministic.
  * Prompt format (--template):
      alpaca    : fastchat 'alpaca' conversation template. Used for the controlled-grid
                  (seed-42 LoRA / QLoRA / FFT) ArtPrompt cells reported in the paper, and
                  shared by all three methods within every contrast.
      tokenizer : the model's own chat template via tokenizer.apply_chat_template, with a
                  leading BOS stripped (as HarmBench get_template() does). This is the format
                  the original-grid HarmBench ArtPrompt runs used, and the one used for the
                  MixAT defended/undefended comparison.

Writes {attack, model, benchmark, behavior_id, behavior_goal, behavior_category, prompt, response}
as JSONL for scoring with hb_cls_score.py / secondary_score.py.

Usage:
  python artprompt_phase5.py --model_path models/phase5/llama31_lora_seed42_merged \
    --test_cases toolkits/HarmBench/results_jbb/ArtPrompt/llama31_lora_custom/test_cases/test_cases.json \
    --behaviors_csv toolkits/HarmBench/data/behavior_datasets/jbb_behaviors.csv \
    --benchmark jbb --template alpaca \
    --out toolkits/HarmBench/results_phase5/artprompt/llama31_lora_seed42_jbb.jsonl
"""
import argparse, json, os, csv, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


def build_prompt_fmt(mode, tokenizer, model_path):
    """Prompt format carrying a literal {instruction} placeholder.

    Note on HarmBench: baselines/model_utils.py get_template() builds a template from the
    fschat template named in models.yaml and then, when no chat_template is set, replaces it
    with tokenizer.apply_chat_template. The original-grid HarmBench runs therefore used the
    tokenizer's chat template ('tokenizer' mode here).
    """
    if mode == "tokenizer":
        msgs = [{"role": "user", "content": "{instruction}"}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # get_template strips a leading BOS (baselines re-add it downstream); mirror that.
        if tokenizer.bos_token and p.startswith(tokenizer.bos_token):
            p = p.replace(tokenizer.bos_token, "", 1)
        return p
    if mode == "alpaca":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("alpaca")
        conv.append_message(conv.roles[0], "{instruction}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    raise ValueError(f"unknown template mode {mode}")


def load_goals(behaviors_csv):
    goals, cats = {}, {}
    with open(behaviors_csv) as f:
        for r in csv.DictReader(f):
            bid = r.get("BehaviorID") or r.get("Index")
            goal = r.get("Behavior") or r.get("Goal")
            if bid and goal:
                goals[str(bid)] = goal
                cats[str(bid)] = r.get("SemanticCategory", r.get("Category", ""))
    return goals, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_cases", required=True, help="original-grid cell test_cases.json (reused)")
    ap.add_argument("--behaviors_csv", required=True)
    ap.add_argument("--benchmark", required=True, choices=["jbb", "harmbench"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    # See the module docstring: 'alpaca' for the controlled-grid cells, 'tokenizer' for the
    # MixAT comparison and for matching the original-grid HarmBench prompt format.
    ap.add_argument("--template", default="tokenizer", choices=["tokenizer", "alpaca"])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}

    tc = json.load(open(a.test_cases))               # {behavior_id: [test_case_str]}
    goals, cats = load_goals(a.behaviors_csv)
    missing = [b for b in tc if str(b) not in goals]
    if missing:
        print(f"WARN: {len(missing)} test-case ids have no goal in CSV (e.g. {missing[:3]})")

    print(f"==> loading merged model {a.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_fmt = build_prompt_fmt(a.template, tokenizer, a.model_path)
    print(f"==> template={a.template}\n--- prompt format ---\n{prompt_fmt!r}\n---")
    model = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16,
                                                 device_map="auto", **tok_kw)
    model.eval()

    print(f"==> {len(tc)} {a.benchmark} ArtPrompt test cases; writing {a.out}")
    open(a.out, "w").close()
    for bid, cases in tqdm(tc.items(), desc=f"ArtPrompt {a.benchmark}"):
        test_case = cases[0] if isinstance(cases, list) else cases
        goal = goals.get(str(bid), "")
        full = prompt_fmt.format(instruction=test_case)
        mi = tokenizer(full, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                out = model.generate(input_ids=mi["input_ids"], attention_mask=mi["attention_mask"],
                                     max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1,
                                     pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0, mi["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            resp = f"ERROR: {e}"
        with open(a.out, "a") as f:
            f.write(json.dumps({"attack": "ArtPrompt", "model": a.model_path,
                                "benchmark": a.benchmark, "behavior_id": bid,
                                "behavior_goal": goal, "behavior_category": cats.get(str(bid), ""),
                                "prompt": test_case, "response": resp}) + "\n")
    print("==> done")


if __name__ == "__main__":
    main()
