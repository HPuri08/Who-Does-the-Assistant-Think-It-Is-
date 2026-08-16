"""
Track 5 runner: collects responses for every (model x condition x item x rep).

Works with any OpenAI-compatible endpoint (OpenRouter, OpenAI, Gemini's
compat endpoint, a local vLLM server). One JSONL row per response.

Usage:
    export API_KEY=sk-...
    export BASE_URL=https://openrouter.ai/api/v1     # or your provider
    python run_experiment.py --models anthropic/claude-3.5-haiku \
        --study all --out data/raw.jsonl

    # dry run (no API calls, prints the request matrix):
    python run_experiment.py --dry-run
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import config

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def build_messages(cond: dict, item_text: str):
    """Construct the chat messages for a condition + item."""
    user = cond["wrap"].format(item=item_text)
    msgs = []
    if cond.get("system"):
        msgs.append({"role": "system", "content": cond["system"]})
    msgs.append({"role": "user", "content": user})
    return msgs


def request_kwargs(model: str):
    """Per-model request settings (see config: some models 400 on temperature,
    some need headroom because thinking shares the max_tokens budget)."""
    kwargs = {"max_tokens": config.max_tokens_for(model)}
    if config.accepts_temperature(model):
        kwargs["temperature"] = config.TEMPERATURE
    return kwargs


def call_model(client, model: str, cond: dict, item_text: str,
               max_retries: int = 4):
    """One API call with basic retry/backoff. Returns response text."""
    msgs = build_messages(cond, item_text)
    kwargs = request_kwargs(model)
    for attempt in range(max_retries):
        try:
            if cond.get("completion_mode"):
                # Raw completion endpoint (needs a base/completions model).
                resp = client.completions.create(
                    model=model,
                    prompt=msgs[-1]["content"],
                    **kwargs,
                )
                return resp.choices[0].text.strip()
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                **kwargs,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"    retry {attempt + 1} in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)
    return None


def call_pushback(client, model: str, cond: dict, item_text: str,
                  first_answer: str):
    """Study 3: send one adversarial pushback turn after I01."""
    msgs = build_messages(cond, item_text)
    msgs.append({"role": "assistant", "content": first_answer})
    msgs.append({"role": "user", "content": config.PUSHBACK})
    try:
        resp = client.chat.completions.create(
            model=model, messages=msgs, **request_kwargs(model),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        print(f"    pushback failed: {e}", file=sys.stderr)
        return None


def iter_jobs(models, studies, include_optional):
    for model in models:
        for cid, cond in config.CONDITIONS.items():
            if cond.get("optional") and not include_optional:
                continue
            batteries = []
            if studies in ("identity", "all"):
                batteries.append(("identity", config.IDENTITY_ITEMS))
            if studies in ("preference", "all"):
                batteries.append(("preference", config.PREFERENCE_ITEMS))
            if studies in ("preservation", "all"):
                batteries.append(("preservation", config.PRESERVATION_ITEMS))
            for study, items in batteries:
                for item in items:
                    for rep in range(config.REPS):
                        yield model, cid, cond, study, item, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODELS_DEFAULT)
    ap.add_argument("--study",
                    choices=["identity", "preference", "preservation", "all"],
                    default="all")
    ap.add_argument("--out", default="data/raw.jsonl")
    ap.add_argument("--include-optional", action="store_true",
                    help="include C4 raw-completion condition")
    ap.add_argument("--pushback", action="store_true",
                    help="run Study 3 adversarial pushback on item I01")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="seconds between calls (rate-limit politeness)")
    args = ap.parse_args()

    jobs = list(iter_jobs(args.models, args.study, args.include_optional))
    print(f"Total calls planned: {len(jobs)}"
          + (" (+ pushback turns)" if args.pushback else ""))

    if args.dry_run:
        for model, cid, _, study, item, rep in jobs[:10]:
            print(f"  {model} | {cid} | {study}/{item['id']} | rep{rep}")
        print(f"  ... and {max(0, len(jobs) - 10)} more")
        return

    if OpenAI is None:
        sys.exit("pip install openai")
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        sys.exit("Set API_KEY env var.")
    client = OpenAI(api_key=api_key, base_url=base_url)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip rows already collected.
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["model"], r["condition"], r["item_id"],
                          r["rep"], r.get("turn", 1)))
        print(f"Resuming: {len(done)} rows already collected.")

    with open(out_path, "a", encoding="utf-8") as f:
        for i, (model, cid, cond, study, item, rep) in enumerate(jobs):
            key = (model, cid, item["id"], rep, 1)
            if key in done:
                continue
            print(f"[{i + 1}/{len(jobs)}] {model} {cid} {item['id']} r{rep}")
            text = call_model(client, model, cond, item["text"])
            row = {
                "model": model, "condition": cid, "arm": cond["arm"],
                "study": study, "item_id": item["id"], "item_cat": item["cat"],
                "item_text": item["text"], "rep": rep, "turn": 1,
                "response": text, "ts": time.time(),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            if (args.pushback and item["id"] == "I01" and text
                    and not cond.get("completion_mode")):
                pb = call_pushback(client, model, cond, item["text"], text)
                row2 = dict(row, turn=2, response=pb,
                            pushback=config.PUSHBACK, ts=time.time())
                f.write(json.dumps(row2) + "\n")
                f.flush()
            time.sleep(args.sleep)

    print(f"Done. Data in {out_path}")


if __name__ == "__main__":
    main()
