import os
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

def gather_text_files(source_dir, suffix=".txt"):
    """Recursively collect all text-file paths under source_dir."""
    text_files = []
    for root, dirs, files in os.walk(source_dir):
        for fname in files:
            if fname.lower().endswith(suffix):
                text_files.append(os.path.join(root, fname))
    return text_files

class MyTextDataset(Dataset):
    """
    Each sample is a path to a single .txt file.
    We'll read it in the collate function for speed and memory efficiency.
    """
    def __init__(self, source_dir):
        self.file_paths = gather_text_files(source_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.file_paths[idx]

def build_prompt(original_report):
    """
    Build Qwen system+user messages for rewriting MIMIC-CXR reports.
    """
    system_msg = """You are an expert medical assistant AI capable of modifying clinical documents to
user specifications. You make minimal changes to the original document to satisfy
user requests. You never add information that is not already directly stated in
the original document.

Extract only two sections from the input radiology report: 'Findings' and 'Impression'. 
If 'Finding' or 'Impression' is None, keep 'None'. 
An Indication section can refer to the History, Indication or Reason for Study sections in the
original report. Remove any information not directly observable from the current
imaging study. For instance, remove any patient demographic data, past medical
history, or comparison to prior images or studies. The generated 'Findings' and
'Impression' sections should not reference any changes based on prior images,
studies, or external knowledge about the patient. For example, paraphrase sentences 
containing words like 'new', 'unchanged', 'increase', 'decrease' such that 
the section is related only to this specific image.
Rewrite such comparisons as a status observation based only on the current image or study.
The output should be sentences for each section.

Also remove deidentified patient information represented by '___'.
"""

    user_msg = (
        f"Rewrite this chest X-ray report:\n\n{original_report}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def collate_fn(batch_file_paths, tokenizer):
    """
    1) For each file path in the batch, read the text.
    2) Build Qwen chat prompts for each.
    3) Tokenize them in a single step.
    """
    text_prompts = []
    raw_paths = []
    for path in batch_file_paths:
        with open(path, "r", encoding="utf-8") as f:
            original_text = f.read().strip()
        msgs = build_prompt(original_text)
        # Convert to single chat prompt string
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        text_prompts.append(prompt_str)
        raw_paths.append(path)

    # Tokenize as a batch
    model_inputs = tokenizer(
        text_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,      # be safe if text is long
        max_length=512       # or smaller if GPU memory is an issue
    )

    # We'll return both the tokenized inputs and the raw file paths
    return model_inputs, raw_paths

import os
from torch.utils.data import DataLoader
from tqdm import tqdm

def paraphrase_batch(batch_inputs, model, tokenizer, max_new_tokens=512):
    """
    Perform generation on a *batch* of tokenized prompts,
    returning a list of paraphrased strings in the same order.
    """
    # Move inputs to the correct device
    batch_inputs = {k: v.to(model.device) for k,v in batch_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **batch_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # deterministic
        )

    # We'll slice out the newly generated tokens for each sample
    # because we used Qwen's chat prompt approach
    paraphrased_list = []
    for i in range(len(batch_inputs["input_ids"])):
        inp_len = batch_inputs["input_ids"][i].shape[0]
        new_ids = output_ids[i][inp_len:]
        text_out = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        paraphrased_list.append(text_out)

    return paraphrased_list

def main():
    model_name = "Qwen/Qwen2.5-14B-Instruct"

    # 1) Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # Qwen uses left padding

    # 2) Create dataset & dataloader
    source_dir = "/home/arpanp/Fed_Vision_Language/report/incomplete"
    target_dir = "/home/arpanp/Fed_Vision_Language/report/qwen_report_cleaned"
    dataset = MyTextDataset(source_dir)

    # We'll set a batch size. Adjust to fit your GPU memory
    batch_size = 64
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # 4) Loop over loader in distributed manner
    # each GPU gets a portion of the data automatically
    for batch_inputs, file_paths in tqdm(loader, desc="Processing", ncols=80):
        # 5) Generate paraphrased text for the entire batch
        paraphrased_list = paraphrase_batch(batch_inputs, model, tokenizer)

        # 6) Save results. Each GPU will do this for its subset
        for path, cleaned in zip(file_paths, paraphrased_list):
            # build parallel output path
            rel_path = os.path.relpath(path, source_dir)
            out_path = os.path.join(target_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write(cleaned)

if __name__ == "__main__":
    # Launch with: accelerate launch my_script.py
    main()
