#!/usr/bin/env python3
import csv, re, os, time, argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import wandb

# ----------------------------- MODEL CONFIG -----------------------------
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_DEVICE   = "cuda:0"
DEFAULT_DTYPE    = torch.bfloat16

INSTRUCTIONS = (
    "You are an expert research assistant tasked with analyzing scientific papers from the TRACS dataset. "
    "Your job is to read the provided fields (title, abstract, body, acknowledgments, grants) and produce structured predictions.\n\n"
    "Tasks:\n"
    "1. Identify the Telescope:\n"
    "- Determine which telescope the paper is about, from the known telescope list in the dataset.\n"
    "- Output exactly one telescope. If no telescope is identifiable, output \"None\".\n\n"
    "2. Classify the Paper’s Usage:\n"
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
    "- Do not invent other telescope names.\n"
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
    ctx_tokens: int = 8192

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, padding_side="left", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True, trust_remote_code=True
        ).to(self.device)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _truncate(self, text: str, budget: int) -> str:
        if not text: return ""
        ids = self.tokenizer(text, add_special_tokens=False, truncation=True,
                             max_length=budget)["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _build_messages(self, row: Dict) -> List[Dict[str,str]]:
        bibcode = row.get("bibcode") or row.get("Id") or "UNKNOWN_ID"
        per = self.ctx_tokens // 4
        return [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content":
                f"Id={bibcode}\n"
                f"Title: {self._truncate(row.get('title') or '', per//3)}\n"
                f"Abstract: {self._truncate(row.get('abstract') or '', per)}\n"
                f"Body: {self._truncate(row.get('body') or '', per)}\n"
                f"Acknowledgments: {self._truncate(row.get('acknowledgments') or '', per//4)}\n"
                f"Grants: {self._truncate(row.get('grants') or '', per//4)}\n"}
        ]

    @torch.inference_mode()
    def _generate(self, msgs):
        enc = self.tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Build input_ids + attention_mask explicitly (pad==eos safe)
        if isinstance(enc, dict):
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                # safest: explicit mask (can't infer when pad==eos)
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = attention_mask.to(self.model.device)
        else:
            input_ids = enc.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,          # deterministic
            # temperature=0.0,        # remove this if do_sample=False (avoids the warning)
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # If you didn't upgrade transformers, uncomment this:
            # use_cache=False,
        )

        gen_ids = out[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


    def _parse_csv(self, text: str):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines: return None
        for ln in lines:
            if ln.lower().startswith("id,"): continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 6: continue
            norm = lambda b: "True" if b.lower() in ("true","t","1","yes") else "False"
            Id = parts[0].strip('"')
            telescope = parts[1].strip('"') or "None"
            return (Id, telescope, norm(parts[2]), norm(parts[3]), norm(parts[4]), norm(parts[5]))
        return None

    def predict_split_to_csv(self, ds_split, out_csv: str, limit=None) -> int:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Id","telescope","science","instrumentation","mention","not_telescope"])
            iterator = ds_split if limit is None else ds_split.select(range(min(limit,len(ds_split))))
            n=0
            for row in tqdm(iterator, desc=f"Predicting → {out_csv}"):
                parsed = self._parse_csv(self._generate(self._build_messages(row)))
                Id = row.get("bibcode","UNKNOWN_ID")
                if parsed: writer.writerow(list(parsed))
                else: writer.writerow([Id,"None","False","False","False","False"])
                n+=1
        print(f"✅ Wrote {n} rows → {out_csv}")
        return n

# ----------------------------- MAIN -----------------------------
def dtype_from_str(s): return {"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}[s]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--wandb_project", default="tracs_llm_runs")
    ap.add_argument("--wandb_run", default=None)
    args = ap.parse_args()

    ds = load_dataset("adsabs/TRACS")
    prefix = args.wandb_run or os.path.basename(args.model_id).replace("/", "_")
    out_dir = os.path.join("outputs", prefix)
    os.makedirs(out_dir, exist_ok=True)

    wandb.init(project=args.wandb_project, name=prefix, config=vars(args))

    tr = TRACS(model_id=args.model_id, device=args.device, torch_dtype=dtype_from_str(args.dtype))
    # splits = ["NONE_reference_test", "JWST_reference_test", "CHANDRA_reference_test", "HST_reference_test"]
    splits = ["HST_reference_test"]
    total = 0
    for split in splits:
        if split not in ds:
            print(f"[WARN] Missing {split}")
            continue
        out_csv = os.path.join(out_dir, f"{prefix}_{split}.csv")
        n = tr.predict_split_to_csv(ds[split], out_csv, args.limit)
        wandb.log({f"{split}/rows": n})
        art = wandb.Artifact(f"{prefix}_{split}", type="predictions")
        art.add_file(out_csv)
        wandb.log_artifact(art)
        total += n
    wandb.summary["total_rows"] = total
    wandb.finish()

if __name__ == "__main__":
    main()
