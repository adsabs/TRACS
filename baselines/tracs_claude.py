#!/usr/bin/env python3
"""
Run Claude on TRACS telescope/reference classification splits.

Example smoke test:
    python tracs_claude.py --limit 2 --save_raw_outputs --no_wandb

Example full run:
    python tracs_claude.py --model_id claude-opus-4-6 --no_wandb

Requires:
    pip install anthropic datasets tqdm wandb
    export ANTHROPIC_API_KEY="..."
"""

import argparse
import csv
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None


# ----------------------------- CONFIG -----------------------------

SPLITS = [
    ("NONE_reference_test",    "claude_NONE_reference_test.csv"),
    ("JWST_reference_test",    "claude_JWST_reference_test.csv"),
    ("CHANDRA_reference_test", "claude_CHANDRA_reference_test.csv"),
    ("HST_reference_test",     "claude_HST_reference_test.csv"),
]

DEFAULT_MODEL_ID = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")

INSTRUCTIONS = (
    "You are an expert research assistant tasked with analyzing scientific papers from the TRACS dataset. "
    "Your job is to read the provided fields and produce structured predictions.\n\n"
    "Tasks:\n"
    "1. Identify the Telescope:\n"
    "- Output exactly one telescope from: CHANDRA, JWST, HST, or None.\n"
    "- If no telescope is identifiable, output None.\n\n"
    "2. Classify the Paper's Usage:\n"
    "- science → True if the paper uses telescope data to produce new scientific results, else False.\n"
    "- instrumentation → True if the paper describes technical or engineering aspects of the telescope, else False.\n"
    "- mention → True if the paper references the telescope but does not generate new results, else False.\n"
    "- not_telescope → True if the reference is unrelated or a false positive, else False.\n"
    "Multiple labels can be True.\n\n"
    "3. Output Format:\n"
    "Return exactly one CSV row and nothing else. Do not include the header.\n"
    "The row must have exactly these six fields:\n"
    "Id,telescope,science,instrumentation,mention,not_telescope\n"
    "Example row:\n"
    "2012A&A...537A..18M_CHANDRA,JWST,True,False,False,False\n\n"
    "Guidelines:\n"
    "- Valid telescope outputs are exactly: CHANDRA, JWST, HST, or None.\n"
    "- If the paper does not clearly refer to CHANDRA, JWST, or HST, output None.\n"
    "- Do not invent other telescope names.\n"
    "- Use only the paper text provided.\n"
    "- If unsure, output None.\n"
    "- Ensure strict CSV compliance with commas between fields and no trailing commas.\n"
    "- Do not include explanations, markdown, reasoning, or extra text."
)


# ----------------------------- HELPERS -----------------------------

def normalize_bool(value: str) -> str:
    v = str(value).strip().strip('"').strip("'").lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return "True"
    if v in {"false", "f", "0", "no", "n"}:
        return "False"
    return "False"


def normalize_telescope(value: str) -> str:
    tel = str(value).strip().strip('"').strip("'")
    upper = tel.upper()
    if upper in {"CHANDRA", "JWST", "HST"}:
        return upper
    if tel.lower() == "none" or tel == "":
        return "None"
    return "None"


def safe_run_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(model_id.strip("/")))


def truncate_chars(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]..."


# ----------------------------- CLAUDE RUNNER -----------------------------

@dataclass
class TRACSClaudeRunner:
    model_id: str = DEFAULT_MODEL_ID
    max_tokens: int = 128
    temperature: float = 0.0
    ctx_chars: int = 24000
    raw_dir: Optional[Path] = None
    debug: bool = False
    api_key: Optional[str] = None
    sleep_seconds: float = 0.0

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY. Set it with: export ANTHROPIC_API_KEY='...'")

        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

        if self.raw_dir is not None:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialized Claude runner with model={self.model_id}", flush=True)

    def _build_prompt_for_paper(self, row: Dict) -> Tuple[str, str]:
        bibcode = row.get("bibcode") or row.get("Id") or row.get("id") or "UNKNOWN_ID"

        # Character budget because Claude API does not need local tokenization.
        # Keep metadata fields, then give body the remainder.
        title = truncate_chars(row.get("title") or "", 2000)
        abstract = truncate_chars(row.get("abstract") or "", 6000)
        ack = truncate_chars(row.get("acknowledgments") or "", 3000)
        grants = truncate_chars(row.get("grants") or "", 3000)

        used = len(title) + len(abstract) + len(ack) + len(grants)
        body_budget = max(1000, self.ctx_chars - used - 2000)
        body = truncate_chars(row.get("body") or "", body_budget)

        user_prompt = (
            f"Id={bibcode}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Body: {body}\n"
            f"Acknowledgments: {ack}\n"
            f"Grants: {grants}\n"
        )

        return str(bibcode), user_prompt

    def _generate_claude(self, user_prompt: str) -> Tuple[str, Dict]:
        start = time.time()

        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=INSTRUCTIONS,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        text = "\n".join(text_parts).strip()

        # Defensive cleanup if Claude ever uses markdown.
        if "```csv" in text:
            text = text.split("```csv", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        usage = getattr(resp, "usage", None)
        tokens = None
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            tokens = {
                "input": input_tokens,
                "output": output_tokens,
                "total": (
                    (input_tokens or 0) + (output_tokens or 0)
                    if input_tokens is not None or output_tokens is not None
                    else None
                ),
            }

        metadata = {
            "response_id": getattr(resp, "id", None),
            "seconds": time.time() - start,
            "tokens": tokens,
        }

        return text, metadata

    def _parse_csv_row(self, text: str, id_fallback: str) -> Tuple[str, str, str, str, str, str]:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]

        header_like = re.compile(
            r"^Id\s*,\s*telescope\s*,\s*science\s*,\s*instrumentation\s*,\s*mention\s*,\s*not_telescope$",
            re.I,
        )

        for ln in lines:
            if header_like.match(ln):
                continue

            ln = ln.strip("` ").lstrip("- ").strip()

            try:
                rows = list(csv.reader(io.StringIO(ln)))
            except Exception:
                rows = []

            if not rows:
                continue

            parts = [p.strip() for p in rows[0]]

            if len(parts) < 6:
                continue

            if len(parts) > 6:
                parts = parts[:5] + [",".join(parts[5:]).strip()]

            out_id = parts[0].strip().strip('"').strip("'") or id_fallback
            telescope = normalize_telescope(parts[1])
            science = normalize_bool(parts[2])
            instrumentation = normalize_bool(parts[3])
            mention = normalize_bool(parts[4])
            not_telescope = normalize_bool(parts[5])

            return (out_id, telescope, science, instrumentation, mention, not_telescope)

        return (id_fallback, "None", "False", "False", "False", "False")

    def predict_split_to_csv(
        self,
        ds_split,
        out_csv_path: str,
        split_name: str,
        limit: Optional[int] = None,
    ) -> int:
        fields = ["Id", "telescope", "science", "instrumentation", "mention", "not_telescope"]
        out_csv_path = Path(out_csv_path)

        iterator = ds_split if limit is None else ds_split.select(range(min(limit, len(ds_split))))

        n_rows = 0
        start = time.time()

        with out_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            f.flush()

            for idx, row in enumerate(tqdm(iterator, desc=f"Predicting → {out_csv_path.name}")):
                row_start = time.time()
                id_fallback, user_prompt = self._build_prompt_for_paper(row)

                try:
                    raw_text, metadata = self._generate_claude(user_prompt)
                    parsed = self._parse_csv_row(raw_text, id_fallback=id_fallback)
                    error = None
                except Exception as e:
                    raw_text = ""
                    metadata = {"seconds": time.time() - row_start}
                    parsed = (id_fallback, "None", "False", "False", "False", "False")
                    error = str(e)
                    print(f"[ERROR] split={split_name} idx={idx} id={id_fallback}: {e}", flush=True)

                writer.writerow(list(parsed))
                f.flush()
                n_rows += 1

                if self.raw_dir is not None:
                    raw_path = self.raw_dir / f"{split_name}_{idx:06d}.json"
                    raw_path.write_text(json.dumps({
                        "split": split_name,
                        "index": idx,
                        "id": id_fallback,
                        "raw_text": raw_text,
                        "parsed": list(parsed),
                        "metadata": metadata,
                        "error": error,
                        "seconds": time.time() - row_start,
                    }, ensure_ascii=False, indent=2))

                if self.debug and idx < 3:
                    print(
                        f"[debug] row={idx} parsed={parsed} "
                        f"time={time.time() - row_start:.2f}s raw={raw_text[:200]!r}",
                        flush=True,
                    )

                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)

        dur = time.time() - start
        print(f"Wrote {n_rows} predictions → {out_csv_path} in {dur:.1f}s", flush=True)
        return n_rows


# ----------------------------- DRIVER -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run Claude on TRACS splits.")

    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--ctx_chars", type=int, default=24000)

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_dir", default=".")
    p.add_argument("--save_raw_outputs", action="store_true")
    p.add_argument("--raw_dir", default="claude_raw_outputs")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--sleep_seconds", type=float, default=0.0)

    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="tracs_llm_runs")
    p.add_argument("--wandb_run", default=None)

    return p.parse_args()


def main():
    args = parse_args()

    print("Loading TRACS dataset...", flush=True)
    ds = load_dataset("adsabs/TRACS")
    print("Loaded TRACS dataset splits:", list(ds.keys()), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / args.raw_dir if args.save_raw_outputs else None

    runner = TRACSClaudeRunner(
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ctx_chars=args.ctx_chars,
        raw_dir=raw_dir,
        debug=args.debug,
        sleep_seconds=args.sleep_seconds,
    )

    use_wandb = (not args.no_wandb) and (wandb is not None)

    if use_wandb:
        run_name = args.wandb_run or f"{safe_run_name(args.model_id)}_{int(time.time())}"
        wandb.init(project=args.wandb_project, name=run_name, config={
            "model_id": args.model_id,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "ctx_chars": args.ctx_chars,
            "limit": args.limit,
            "splits": [s for s, _ in SPLITS],
        })
    elif not args.no_wandb and wandb is None:
        print("wandb is not installed; continuing without W&B.", flush=True)

    total_rows = 0

    for split_name, out_csv in SPLITS:
        if split_name not in ds:
            print(f"[WARN] Split '{split_name}' not found; skipping.", flush=True)
            continue

        out_path = output_dir / out_csv
        print(f"Using split: {split_name} → {out_path}", flush=True)

        n = runner.predict_split_to_csv(
            ds_split=ds[split_name],
            out_csv_path=str(out_path),
            split_name=split_name,
            limit=args.limit,
        )

        total_rows += n

        if use_wandb:
            art = wandb.Artifact(name=f"claude_predictions_{split_name}", type="predictions")
            art.add_file(str(out_path))
            wandb.log({f"{split_name}/rows": n})
            wandb.log_artifact(art)

    if use_wandb:
        wandb.summary["total_rows"] = total_rows
        wandb.finish()

    print(f"Done. total_rows={total_rows}", flush=True)


if __name__ == "__main__":
    main()