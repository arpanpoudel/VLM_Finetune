import os
import torch
import gradio as gr
from PIL import Image
from transformers import Qwen2_5_VLProcessor
from peft import PeftModel
from models import load_qwen_model
from qwen_vl_utils import process_vision_info
from absl import flags, app
from ml_collections.config_flags import config_flags
from functools import partial

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    'configs/base_config.py',
    'Path to the config file.',
    lock_config=True
)

# Define the system prompt
system_message = """You are an expert radiologist trained in interpreting chest X-rays. Given radiographic image of the chest (frontal view), generate a detailed and clinically accurate radiology report. The report should include a summary of the observed findings and, if possible, an impression that highlights key diagnoses or abnormalities. Do not speculate beyond the image content. Use professional radiological language. Maintain a neutral, factual tone suitable for inclusion in a patient medical record. Do NOT reference previous imaging or mention paging physicians. Instead, describe the chest x-ray in a stand-alone format."""

# Format data function
def format_data(image):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]

# Generate report function
def generate_report(image, processor, model):
    sample = format_data(image)
    text_input = processor.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(sample)
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)

    trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]

# Gradio interface function
def app_interface(image_input, processor, model):
    return generate_report(image_input, processor, model)

# Load model and processor globally for efficiency
def load_model_and_processor(config):
    config.peft.USE_LORA = False
    model = load_qwen_model(config)
    model = PeftModel.from_pretrained(model, config.peft.trained_model_path, dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained(**config.processor)
    processor.tokenizer.padding_side = "right"
    model.eval()
    return model, processor

def main(argv):
    config = FLAGS.config
    model, processor = load_model_and_processor(config)

    # Use partial to pass processor and model to Gradio interface
    interface_fn = partial(app_interface, processor=processor, model=model)

    # Launch Gradio app
    gr.Interface(
        fn=interface_fn,
        inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
        outputs=gr.Textbox(label="Generated Radiology Report"),
        title="Chest X-ray Report Generator",
        description="Upload a chest X-ray image to generate a clinical radiology report"
    ).launch(share=True)

if __name__ == "__main__":
    app.run(main)
