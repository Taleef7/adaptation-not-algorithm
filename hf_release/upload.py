"""Upload the released checkpoints to the Hugging Face Hub.

Usage (after `hf auth login` in your own terminal):
    python upload.py --dry-run          # show what would be uploaded
    python upload.py                    # upload all eight (resumable; safe to re-run)
    python upload.py --only phi-4-Alpaca-FFT

Each repo is created as a gated model (automatic approval after the user accepts the
access terms), so weights are public but downloads require a Hub account and consent.
"""
import argparse, json, os, shutil, sys
from huggingface_hub import HfApi

RESEARCH = os.environ.get("RESEARCH_REPO", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "stage")
KEEP = (".safetensors", ".json", ".jinja", ".model", ".txt")


def stage(m):
    src = os.path.join(RESEARCH, m["src"])
    dst = os.path.join(STAGE, m["repo"])
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if os.path.isfile(p) and f.endswith(KEEP) and f != "training_args.bin":
            os.symlink(p, os.path.join(dst, f))
    for f in os.listdir(os.path.join(HERE, "cards", m["repo"])):
        shutil.copy(os.path.join(HERE, "cards", m["repo"], f), dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    for m in json.load(open(os.path.join(HERE, "manifest.json"))):
        if args.only and m["repo"] != args.only:
            continue
        repo_id = f"{user}/{m['repo']}"
        dst = stage(m)
        size = sum(os.path.getsize(os.path.realpath(os.path.join(dst, f))) for f in os.listdir(dst))
        print(f"{repo_id}: {len(os.listdir(dst))} files, {size / 1e9:.1f} GB from {m['src']}")
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.update_repo_settings(repo_id, gated="auto")
        api.upload_large_folder(repo_id, dst, repo_type="model", num_workers=4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
