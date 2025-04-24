import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import PeftModel
from models import load_qwen_model
from qwen_vl_utils import process_vision_info
from PIL import Image

def format_data(sample, system_message):
        return [
            {
                "role":"system",
                "content" : [{"type": "text", "text": system_message}]
            },
            {
                "role":"user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    }
                ]
            },
            {
                "role": "assistant",
                "content":[
                    {"type": "text", "text": sample["text"]},
                ]
            }
        ]
        
def test(config, image_path):
    #load processor
    processor = Qwen2_5_VLProcessor.from_pretrained(**config.processor)
    #build base  model
    config.peft.USE_LORA = False
    config.peft.USE_QLORA = False
    model = load_qwen_model(config)
    
    #load peft model
    model = PeftModel.from_pretrained(model, config.peft.lora_adapter_path, dtype =torch.bfloat16)
    model = model.merge_and_unload()
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    text = None
    sample = {
        "image": image,
        "text": text
    }
    
    system_message ="""You are an expert radiologist trained in interpreting chest X-rays. Given one or two radiographic images of the chest (frontal and/or lateral views), generate a detailed and clinically accurate radiology report. The report should include a structured summary of the observed findings and, if possible, an impression that highlights key diagnoses or abnormalities. Do not speculate beyond the image content. If no abnormality is present, state "No acute cardiopulmonary process." Use professional radiological language, and focus on anatomical structures such as lungs, heart, mediastinum, pleura, bones, and devices (e.g., lines, tubes). Maintain a neutral, factual tone suitable for inclusion in a patient medical record."""
    # Format the data
    formatted_sample = format_data(sample, system_message)
    # Generate text from the sample
    output_text = generate_text_from_sample(model, processor, formatted_sample)
    print("Generated Text:")
    print(output_text)

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[:2], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]