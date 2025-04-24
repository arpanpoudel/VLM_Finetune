import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen2.5-14B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def build_paraphrase_prompt(original_report):
    """
    Build Qwen system+user messages for rewriting MIMIC-CXR reports.
    """
    system_msg = """ You are an expert medical assistant AI capable of modifying clinical documents to
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
        f"Rewrite this chest X-ray report\n\n{original_report}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def paraphrase_mimic_report(original_report, max_new_tokens=512):
    """
    Use Qwen to paraphrase the MIMIC-CXR report.
    """
    messages = build_paraphrase_prompt(original_report)
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed_ids = [
        out[len(inp) :] for inp, out in zip(model_inputs.input_ids, output_ids)
    ]
    paraphrased = tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
    return paraphrased.strip()

def gather_text_files(source_dir):
    """
    Recursively collects all .txt file paths under `source_dir`.
    Returns a list of file paths.
    """
    text_files = []
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(".txt"):
                text_files.append(os.path.join(root, filename))
    return text_files

def process_all_files(source_dir, target_dir):
    all_txt_paths = gather_text_files(source_dir)

    for source_path in tqdm(all_txt_paths, desc="Processing files", ncols=80):
        # Build the corresponding output path in 'target_dir'
        rel_path = os.path.relpath(source_path, source_dir)
        target_path = os.path.join(target_dir, rel_path)
        
        # Ensure the subdirectory structure exists in 'target_dir'
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Read file content
        with open(source_path, "r", encoding="utf-8") as f:
            original_text = f.read().strip()

        # Clean / paraphrase
        cleaned_text = paraphrase_mimic_report(original_text)

        # Write cleaned text
        with open(target_path, "w", encoding="utf-8") as out_f:
            out_f.write(cleaned_text)

def main():
    source_dir = "/home/arpanp/Fed_Vision_Language/report/incomplete"  
    target_dir = "/home/arpanp/Fed_Vision_Language/report/qwen_report_cleaned"       
    process_all_files(source_dir, target_dir)

if __name__ == "__main__":
    main()
