#!/usr/bin/env python3
import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb  # optional; not used below but kept for your env

# ----------------------------- CONFIG -----------------------------
# Set your four TRACS splits here (name, output_csv)
SPLITS = [
    ("NONE_reference_test", "mistral_NONE_reference_test.csv"),
    ("JWST_reference_test",  "mistral_JWST_reference_test.csv"),
    ("CHANDRA_reference_test","mistral_pred_CHANDRA_reference_test.csv"),
    ("HST_reference_test",  "mistral_HST_reference_test.csv"),
]
# Choose your Mistral instruct model
DEFAULT_MODEL_ID = "teknium/OpenHermes-2.5-Mistral-7B"  # or "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_DEVICE   = "cuda:7"  # pick your GPU
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
    "- If the paper does not clearly refer to CHANDRA, JWST, or HST, output None."
    "- Do not invent other telescope names."
    "- Use only the paper’s text for decisions.\n"
    "- If unsure about the telescope, output \"None\".\n"
    "- Ensure strict CSV compliance (commas between fields, no trailing commas).\n"
    "- Do not include explanations or reasoning in the final output."
)

# ----------------------------- MODEL WRAPPER -----------------------------
@dataclass
class TRACS:
    model_id: str
    max_new_tokens: int = 256            # enough for a single CSV row reliably
    device: str = DEFAULT_DEVICE         # e.g., "cuda:2" or "cpu"
    torch_dtype: torch.dtype = DEFAULT_DTYPE
    ctx_tokens: int = 8192               # prompt token headroom (adjust to VRAM)

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, padding_side="left", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True,
            device_map=None, trust_remote_code=True
        ).to(self.device)

        # Try to enable flash attention if supported
        try:
            self.model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

    # ------- token-aware truncation -------
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

    # ------- prompt assembly for a single paper -------
    def _build_messages_for_paper(self, row: Dict) -> List[Dict[str, str]]:
        bibcode = row.get("bibcode") or row.get("Id") or "UNKNOWN_ID"

        # Allocate token budget across fields; leave room for system + output
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

        # Mistral instruct models expect system+user messages
        return [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

    # ------- generation with Mistral chat template -------
    @torch.inference_mode()
    def _generate_chat(self, messages: List[Dict[str, str]]) -> str:
        # Render messages to a single prompt using the model's chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,             # greedy for determinism
            temperature=0.0,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Remove the prompt tokens
        gen_ids = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ------- CSV parsing (unchanged) -------
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

    # ------- main loop for one split -------
    def predict_split_to_csv(
        self,
        ds_split,
        out_csv_path: str,
        limit: Optional[int] = None,
        batch_size: int = 1,   # keep 1 for chat models
    ) -> None:
        fields = ["Id", "telescope", "science", "instrumentation", "mention", "not_telescope"]
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

            iterator = ds_split if limit is None else ds_split.select(range(min(limit, len(ds_split))))

            for row in tqdm(iterator, desc=f"Predicting → {out_csv_path}"):
                msgs = self._build_messages_for_paper(row)
                text = self._generate_chat(msgs)
                parsed = self._parse_csv_row(text)

                # fallback Id
                id_fallback = "UNKNOWN_ID"
                try:
                    # recover Id from the constructed user prompt
                    id_fallback = msgs[1]["content"].split("Id=")[1].split("\n", 1)[0].strip()
                except Exception:
                    pass

                if parsed is None:
                    writer.writerow([id_fallback, "None", "False", "False", "False", "False"])
                else:
                    writer.writerow(list(parsed))

        print(f"Wrote predictions → {out_csv_path}")

# ----------------------------- DRIVER -----------------------------
def main():
    ds = load_dataset("adsabs/TRACS")
    print("Loaded TRACS dataset splits:", list(ds.keys()))

    tr = TRACS(
        model_id=DEFAULT_MODEL_ID,
        max_new_tokens=256,       # a single CSV line; bump if needed
        device=DEFAULT_DEVICE,
        torch_dtype=DEFAULT_DTYPE,
        ctx_tokens=8192,
    )

    # Set limit=None to run full splits
    LIMIT = None  # e.g., 10 for smoke test

    for split_name, out_csv in SPLITS:
        if split_name not in ds:
            print(f"[WARN] Split '{split_name}' not found in dataset; skipping.")
            continue
        print(f"Using split: {split_name} → {out_csv}")
        tr.predict_split_to_csv(
            ds_split=ds[split_name],
            out_csv_path=out_csv,
            limit=LIMIT,
            batch_size=1,
        )

if __name__ == "__main__":
    main()
