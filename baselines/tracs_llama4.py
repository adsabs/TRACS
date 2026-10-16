#!/usr/bin/env python3
import csv
import re
import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
from tqdm import tqdm
import wandb

# ----------------------------- CONFIG -----------------------------
SPLITS = [
    ("NONE_reference_test",    "llama4_NONE_reference_test.csv"),
    ("JWST_reference_test",    "llama4_JWST_reference_test.csv"),
    ("CHANDRA_reference_test", "llama4_CHANDRA_reference_test.csv"),
    ("HST_reference_test",     "llama4_HST_reference_test.csv"),
]

DEFAULT_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
DEFAULT_DEVICE   = "cuda:0"
DEFAULT_DTYPE    = torch.bfloat16

INSTRUCTIONS = (
    "You are an expert research assistant tasked with analyzing scientific papers from the TRACS dataset. "
    "Your job is to read the provided fields (title, abstract, body, acknowledgments, grants) and produce structured predictions.\n\n"
    "Tasks:\n"
    "1. Identify the Telescope:\n"
    "- Determine which telescope the paper is about, from the known telescope list in the dataset.\n"
    "- Output exactly one telescope. If no telescope is identifiable, output \"None\".\n\n"
    "2. Classify the Paper's Usage:\n"
    "- science → True if the paper uses telescope data to produce new results, else False.\n"
    "- instrumentation → True if the paper describes technical or engineering aspects of the telescope, else False.\n"
    "- mention → True if the paper references the telescope but does not generate new results, else False.\n"
    "- not_telescope → True if the reference is unrelated or a false positive, else False.\n"
    "Multiple labels can be True.\n\n"
    "3. Output Format:\n"
    "Always respond in strict CSV-row format with no extra commentary:\n"
    "Id,telescope,science,instrumentation,mention,not_telescope\n"
    "Example:\n"
    "2012A&A...537A..18M_CHANDRA,JWST,True,False,False,False\n\n"
    "Guidelines:\n"
    "- Valid telescope outputs are exactly one of: CHANDRA, JWST, HST, or None. "
    "- If the paper does not clearly refer to CHANDRA, JWST, or HST, output None."
    "- Do not invent other telescope names."
    "- Use only the paper's text for decisions.\n"
    "- If unsure about the telescope, output \"None\".\n"
    "- Ensure strict CSV compliance (commas between fields, no trailing commas).\n"
    "- Do not include explanations or reasoning in the final output."
)

# ----------------------------- MODEL WRAPPER -----------------------------
@dataclass
class TRACS:
    model_id: str
    max_new_tokens: int = 256
    device: str = DEFAULT_DEVICE
    torch_dtype: torch.dtype = DEFAULT_DTYPE
    ctx_tokens: int = 32768  # Llama 4 Scout supports up to 10M tokens

    def __post_init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                use_fast=True,
                padding_side="left",
                trust_remote_code=True,
            )
        except Exception as e:
            self._hint_on_hf_error(e, "tokenizer")

        try:
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            self._hint_on_hf_error(e, "model")

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _hint_on_hf_error(self, e: Exception, stage: str):
        msg = str(e)
        print(f"\n[HF ERROR during {stage}] {msg}\n")
        if "gated repo" in msg.lower() or "403" in msg or "Unauthorized" in msg:
            print(
                "It looks like this model is gated or your account lacks access.\n"
                "Try:\n"
                "  1) `uv run hf auth whoami`  # verify you're logged in\n"
                "  2) Request access on the model page and accept its license\n"
                "  3) Or run with a public instruct model, e.g.:\n"
                "     --model_id HuggingFaceH4/zephyr-7b-beta\n"
            )
        raise

    def _truncate_to_tokens(self, text: str, budget: int) -> str:
        if not text:
            return ""
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=budget,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _build_messages_for_paper(self, row: Dict) -> List[Dict[str, str]]:
        bibcode = row.get("bibcode") or row.get("Id") or "UNKNOWN_ID"

        per = self.ctx_tokens // 4
        title = self._truncate_to_tokens(row.get("title") or "", per // 3)
        abstract = self._truncate_to_tokens(row.get("abstract") or "", per)
        body = self._truncate_to_tokens(row.get("body") or "", per)
        ack = self._truncate_to_tokens(row.get("acknowledgments") or "", per // 4)
        grants = self._truncate_to_tokens(row.get("grants") or "", per // 4)

        user_prompt = (
            f"Id={bibcode}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Body: {body}\n"
            f"Acknowledgments: {ack}\n"
            f"Grants: {grants}\n"
        )
        return [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

    @torch.inference_mode()
    def _generate_chat(self, messages: List[Dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        text = self.tokenizer.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True,
        )[0].strip()

        if text.lower().startswith("assistant"):
            text = text.split("\n", 1)[-1].strip()

        return text

    def _parse_csv_row(self, text: str) -> Optional[Tuple[str, str, str, str, str, str]]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        header_like = re.compile(
            r"^Id\s*,\s*telescope\s*,\s*science\s*,\s*instrumentation\s*,\s*mention\s*,\s*not_telescope$",
            re.I,
        )
        for ln in lines:
            if header_like.match(ln):
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != 6:
                if len(parts) > 6:
                    parts = parts[:5] + [",".join(parts[5:]).strip()]
                else:
                    continue

            def norm_bool(b: str) -> str:
                b = b.strip().strip('"').strip("'")
                if b.lower() in ("true", "t", "1", "yes"): return "True"
                if b.lower() in ("false", "f", "0", "no"): return "False"
                return "False"

            Id = parts[0].strip().strip('"')
            telescope = parts[1].strip().strip('"') or "None"
            science = norm_bool(parts[2])
            instrumentation = norm_bool(parts[3])
            mention = norm_bool(parts[4])
            not_telescope = norm_bool(parts[5])
            return (Id, telescope, science, instrumentation, mention, not_telescope)
        return None

    def predict_split_to_csv(
        self,
        ds_split,
        out_csv_path: str,
        limit: Optional[int] = None,
    ) -> int:
        fields = ["Id", "telescope", "science", "instrumentation", "mention", "not_telescope"]
        n_rows = 0
        start = time.time()
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            iterator = ds_split if limit is None else ds_split.select(range(min(limit, len(ds_split))))
            for row in tqdm(iterator, desc=f"Predicting → {out_csv_path}"):
                msgs = self._build_messages_for_paper(row)
                text = self._generate_chat(msgs)
                parsed = self._parse_csv_row(text)
                try:
                    id_fallback = msgs[1]["content"].split("Id=")[1].split("\n", 1)[0].strip()
                except Exception:
                    id_fallback = "UNKNOWN_ID"

                if parsed is None:
                    writer.writerow([id_fallback, "None", "False", "False", "False", "False"])
                else:
                    writer.writerow(list(parsed))
                n_rows += 1
        dur = time.time() - start
        print(f"Wrote predictions → {out_csv_path} in {dur:.1f}s")
        return n_rows

# ----------------------------- DRIVER -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU indices to expose, e.g. '0,1' or '4'. "
                        "Sets CUDA_VISIBLE_DEVICES before any CUDA init.")
    p.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    p.add_argument("--ctx_tokens", type=int, default=32768)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--wandb_project", default="tracs_llm_runs")
    p.add_argument("--wandb_run", default=None)
    return p.parse_args()

def dtype_from_str(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]

def main():
    args = parse_args()
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"CUDA_VISIBLE_DEVICES={args.gpus}")
    ds = load_dataset("adsabs/TRACS")
    print("Loaded TRACS dataset splits:", list(ds.keys()))

    tr = TRACS(
        model_id=args.model_id,
        max_new_tokens=256,
        device=args.device,
        torch_dtype=dtype_from_str(args.dtype),
        ctx_tokens=args.ctx_tokens,
    )

    run_name = args.wandb_run or f"{os.path.basename(args.model_id)}_{int(time.time())}"
    wandb.init(project=args.wandb_project, name=run_name, config={
        "model_id": args.model_id,
        "device": args.device,
        "gpus": args.gpus,
        "dtype": args.dtype,
        "ctx_tokens": args.ctx_tokens,
        "limit": args.limit,
        "splits": [s for s, _ in SPLITS],
    })

    total_rows = 0
    for split_name, out_csv in SPLITS:
        if split_name not in ds:
            print(f"[WARN] Split '{split_name}' not found; skipping.")
            continue
        print(f"Using split: {split_name} → {out_csv}")
        n = tr.predict_split_to_csv(
            ds_split=ds[split_name],
            out_csv_path=out_csv,
            limit=args.limit,
        )
        total_rows += n

        art = wandb.Artifact(name=f"predictions_{split_name}", type="predictions")
        art.add_file(out_csv)
        wandb.log({f"{split_name}/rows": n})
        wandb.log_artifact(art)

    wandb.summary["total_rows"] = total_rows
    wandb.finish()

if __name__ == "__main__":
    main()
