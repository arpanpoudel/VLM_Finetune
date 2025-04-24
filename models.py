import torch

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_bits_and_bytes_config():
    """
    Get the BitsAndBytesConfig for loading the model with 8-bit quantization.
    
    Returns:
        BitsAndBytesConfig: Configuration for 8-bit quantization.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def get_lora_config(config):
    """
    Get the LoRA configuration for loading the model with LoRA.
    
    Args:
        config: Configuration object containing LoRA parameters.
    
    Returns:
        LoraConfig: Configuration for LoRA.
    """
    target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # all attention projections
    "gate_proj", "up_proj", "down_proj",     # MLP (optional)
    "qkv", "proj"                             # visual attention
    ]
    lora_config = LoraConfig(
        r=config.peft.lora_r,
        lora_alpha=config.peft.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.peft.lora_dropout,
        bias=config.peft.lora_bias,
        task_type=config.peft.task_type,
    )
    return lora_config

def load_qwen_model(config):
    """
    Load the Qwen model from the specified path.
    
    Args:
        model_name_or_path (str): Path to the Qwen model directory or model name.
    
    Returns:
        model: Loaded Qwen model.
    """
    # Load the Qwen model
    if config.peft.USE_QLORA:
        bnb_config = get_bits_and_bytes_config()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model.pretrained_model_name_or_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        if config.peft.USE_LORA:
            lora_config = get_lora_config(config)
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = get_peft_model(model, lora_config)
    elif config.peft.USE_LORA:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model.pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.float16)
        lora_config = get_lora_config(config)
        # Prepare model for PEFT (adds gradient checkpointing, etc.)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model.pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.float16)
    return model