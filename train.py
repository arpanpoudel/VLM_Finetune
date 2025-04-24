import os
import torch
from dataset.dataset import ChestDataset
from dataset.dataset import QwenDataCollator
from transformers import Qwen2_5_VLProcessor, set_seed
from models import load_qwen_model
import wandb
from trl import SFTConfig, SFTTrainer


def train(config):
    
    # Set the random seed for reproducibility
    set_seed(config.training.seed)
    os.environ['PYTHONHASHSEED'] = str(config.training.seed)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    train_dataset = ChestDataset(config.data.train_index_file)
    val_dataset = ChestDataset(config.data.val_index_file)
    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(val_dataset)} validation samples.")
    
    #load processor
    processor = Qwen2_5_VLProcessor.from_pretrained(**config.processor)
    
    #just to make sure
    processor.tokenizer.padding_side = "right"
    
    # Create the data collator
    collator = QwenDataCollator(processor)
    
    #build model
    model = load_qwen_model(config)
    #training args
    training_args = SFTConfig( 
    output_dir=config.training.output_dir,
    num_train_epochs=config.training.num_epochs,
    per_device_train_batch_size=config.training.per_device_batch_size,
    per_device_eval_batch_size=config.training.per_device_eval_batch_size,
    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=config.training.optim,
    learning_rate=config.training.learning_rate,
    lr_scheduler_type=config.training.lr_scheduler_type,
    logging_dir=config.training.logging_dir,
    logging_steps=config.training.logging_steps,
    eval_steps=config.training.eval_steps,
    eval_strategy=config.training.evaluation_strategy,  
    save_steps=config.training.save_steps,
    save_total_limit=config.training.save_total_limit,
    save_strategy=config.training.save_strategy,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=config.training.load_best_model_at_end,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=False,
    report_to="wandb",
    remove_unused_columns=False,
    label_names=["labels"],
    dataset_kwargs={"skip_prepare_dataset":True}
    )
    # Initialize the wandb
    wandb.init(
        project="Qwen-2.5-3b-VL-Finetune",
        config=training_args,
        name="MIMIC-CXR-Finetune_2025-04-12",
        save_code= False,
    )
    

    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )
    # Start training
    trainer.train(resume_from_checkpoint=False)
    # Save the model
    trainer.save_model(config.training.output_dir)
    wandb.finish()
    print("Training completed and model saved.")
        
        
    
    
    
    
  
    
    

    