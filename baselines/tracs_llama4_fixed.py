#!/usr/bin/env python3
"""
Run Llama 4 Scout on TRACS telescope/reference classification splits.

Key fixes vs the earlier version:
- Uses AutoProcessor + Llama4ForConditionalGeneration, matching working Llama 4 scripts.
- Uses Llama 4 text content blocks in apply_chat_template.
- Defaults to smaller context and max_new_tokens=64 so the first row does not stall forever.
- Flushes CSV after each row.
- Optional --save_raw_outputs for debugging model text.
- Optional --no_wandb for local/debug runs.

Example smoke test:
    CUDA_VISIBLE_DEVICES=0,1 python tracs_llama4_fixed.py --limit 1 --ctx_tokens 2048 --no_wandb --save_raw_outputs

Example full run:
    CUDA_VISIBLE_DEVICES=0,1 python tracs_llama4_fixed.py --ctx_tokens 4096 --no_wandb
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
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Llama4ForConditionalGeneration

try:
    import wandb
except Exception:
    wandb = None

# ----------------------------- CONFIG -----------------------------
SPLITS = [
    ("NONE_reference_test",    "llama4_NONE_reference_test.csv"),
    ("JWST_reference_test",    "llama4_JWST_reference_test.csv"),
    ("CHANDRA_reference_test", "llama4_CHANDRA_reference_test.csv"),
    ("HST_reference_test",     "llama4_HST_reference_test.csv"),
]

DEFAULT_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
DEFAULT_DTYPE = torch.bfloat16
VALID_TELESCOPES = {"CHANDRA", "JWST", "HST", "None"}

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
def dtype_from_str(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


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


# ----------------------------- MODEL WRAPPER -----------------------------
@dataclass
class TRACSRunner:
    model_id: str
    max_new_tokens: int = 64
    torch_dtype: torch.dtype = DEFAULT_DTYPE
    ctx_tokens: int = 4096
    raw_dir: Optional[Path] = None
    debug: bool = False

    def __post_init__(self):
        print(f"\nLoading processor: {self.model_id}", flush=True)
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
        except Exception as e:
            self._hint_on_hf_error(e, "processor")

        print(f"Loading model: {self.model_id}", flush=True)
        print(f"Loading model: {self.model_id}", flush=True)
        try:
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            print("Model hf_device_map:", getattr(self.model, "hf_device_map", None), flush=True)
        except Exception as e:
            self._hint_on_hf_error(e, "model")

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if self.raw_dir is not None:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        print("Model loaded. device_map='auto' is active.\n", flush=True)

    def _hint_on_hf_error(self, e: Exception, stage: str):
        msg = str(e)
        print(f"\n[HF ERROR during {stage}] {msg}\n", flush=True)
        if "gated repo" in msg.lower() or "403" in msg or "Unauthorized" in msg:
            print(
                "This model may be gated or your account may lack access. Try:\n"
                "  huggingface-cli login\n"
                "  huggingface-cli whoami\n"
                "Then confirm you accepted the model license on Hugging Face.\n",
                flush=True,
            )
        raise

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _truncate_to_tokens(self, text: str, budget: int) -> str:
        if not text or budget <= 0:
            return ""
        ids = self.tokenizer(
            str(text),
            add_special_tokens=False,
            truncation=True,
            max_length=budget,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _build_messages_for_paper(self, row: Dict) -> Tuple[str, List[Dict[str, str]]]:
        bibcode = row.get("bibcode") or row.get("Id") or row.get("id") or "UNKNOWN_ID"

        # Allocate a bounded token budget. The body gets the remainder after shorter metadata fields.
        total = max(512, int(self.ctx_tokens))
        title_budget = min(256, max(64, total // 16))
        abstract_budget = min(1024, max(128, total // 4))
        ack_budget = min(512, max(64, total // 12))
        grants_budget = min(512, max(64, total // 12))
        body_budget = max(0, total - title_budget - abstract_budget - ack_budget - grants_budget - 512)

        title = self._truncate_to_tokens(row.get("title") or "", title_budget)
        abstract = self._truncate_to_tokens(row.get("abstract") or "", abstract_budget)
        body = self._truncate_to_tokens(row.get("body") or "", body_budget)
        ack = self._truncate_to_tokens(row.get("acknowledgments") or "", ack_budget)
        grants = self._truncate_to_tokens(row.get("grants") or "", grants_budget)

        user_prompt = (
            f"Id={bibcode}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Body: {body}\n"
            f"Acknowledgments: {ack}\n"
            f"Grants: {grants}\n"
        )
        messages = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]
        return str(bibcode), messages

    def _count_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        prompt = "\n".join(m["content"] for m in messages)
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    @torch.inference_mode()
    def _generate_chat(self, messages: List[Dict[str, str]]) -> str:
        llama_messages = [
            {
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}],
            }
            for m in messages
        ]

        inputs = self.processor.apply_chat_template(
            llama_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        text = self.processor.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True,
        )[0].strip()

        if text.lower().startswith("assistant"):
            text = text.split("\n", 1)[-1].strip()
        return text

    def _parse_csv_row(self, text: str, id_fallback: str) -> Tuple[str, str, str, str, str, str]:
        """Parse a one-line CSV response. Fall back to all-False/None when invalid."""
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        header_like = re.compile(
            r"^Id\s*,\s*telescope\s*,\s*science\s*,\s*instrumentation\s*,\s*mention\s*,\s*not_telescope$",
            re.I,
        )

        for ln in lines:
            if header_like.match(ln):
                continue
            # Strip markdown fences or accidental bullets.
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
        n_rows = 0
        start = time.time()

        iterator = ds_split if limit is None else ds_split.select(range(min(limit, len(ds_split))))

        with out_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            f.flush()

            for idx, row in enumerate(tqdm(iterator, desc=f"Predicting → {out_csv_path.name}")):
                id_fallback, msgs = self._build_messages_for_paper(row)

                if self.debug and idx == 0:
                    toks = self._count_prompt_tokens(msgs)
                    print(f"[debug] first prompt id={id_fallback} approx_tokens={toks}", flush=True)

                row_start = time.time()
                raw_text = self._generate_chat(msgs)
                parsed = self._parse_csv_row(raw_text, id_fallback=id_fallback)
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
                        "seconds": time.time() - row_start,
                    }, ensure_ascii=False, indent=2))

                if self.debug and idx < 3:
                    print(f"[debug] row {idx} parsed={parsed} time={time.time() - row_start:.2f}s raw={raw_text[:200]!r}", flush=True)

        dur = time.time() - start
        print(f"Wrote {n_rows} predictions → {out_csv_path} in {dur:.1f}s", flush=True)
        return n_rows


# ----------------------------- DRIVER -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run Llama 4 Scout on TRACS splits.")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU indices to expose, e.g. '0,1'. Prefer setting CUDA_VISIBLE_DEVICES in the shell.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--ctx_tokens", type=int, default=4096,
                   help="Approx token budget for paper fields. Start with 2048 or 4096; 32768 can be very slow.")
    p.add_argument("--max_new_tokens", type=int, default=64,
                   help="Output token budget. One CSV row usually needs <40 tokens.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_dir", default=".")
    p.add_argument("--save_raw_outputs", action="store_true")
    p.add_argument("--raw_dir", default="llama4_raw_outputs")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="tracs_llm_runs")
    p.add_argument("--wandb_run", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

    print("Loading TRACS dataset...", flush=True)
    ds = load_dataset("adsabs/TRACS")
    print("Loaded TRACS dataset splits:", list(ds.keys()), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / args.raw_dir if args.save_raw_outputs else None

    runner = TRACSRunner(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=dtype_from_str(args.dtype),
        ctx_tokens=args.ctx_tokens,
        raw_dir=raw_dir,
        debug=args.debug,
    )

    use_wandb = (not args.no_wandb) and (wandb is not None)
    if use_wandb:
        run_name = args.wandb_run or f"{safe_run_name(args.model_id)}_{int(time.time())}"
        wandb.init(project=args.wandb_project, name=run_name, config={
            "model_id": args.model_id,
            "gpus": args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dtype": args.dtype,
            "ctx_tokens": args.ctx_tokens,
            "max_new_tokens": args.max_new_tokens,
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
            art = wandb.Artifact(name=f"predictions_{split_name}", type="predictions")
            art.add_file(str(out_path))
            wandb.log({f"{split_name}/rows": n})
            wandb.log_artifact(art)

    if use_wandb:
        wandb.summary["total_rows"] = total_rows
        wandb.finish()

    print(f"Done. total_rows={total_rows}", flush=True)


if __name__ == "__main__":
    main()
