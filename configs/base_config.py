import ml_collections

def get_config():
    """Returns the default configuration."""
    config = ml_collections.ConfigDict()
    
    # trainining configurations
    config.training = training = ml_collections.ConfigDict()
    training.output_dir = "./output_3b"
    training.num_epochs = 3
    training.per_device_batch_size = 6
    training.per_device_eval_batch_size = 6
    training.gradient_accumulation_steps = 2
    training.optim="adamw_torch"
    training.learning_rate = 1e-5
    training.lr_scheduler_type = "cosine"
    
    training.logging_dir = "./logs"
    training.logging_steps = 20
    training.eval_steps = 500
    training.evaluation_strategy = "steps"
    training.save_steps = 500
    training.save_strategy = "steps"
    training.load_best_model_at_end = True
    #additional arguments for trainer
    training.save_total_limit = 2
    training.seed = 42
    
    #data configurations
    config.data = data = ml_collections.ConfigDict()
    data.dataset_name = "mimic-cxr"
    data.train_index_file="index_files/mimic_index_frontal_train.json"
    data.val_index_file="index_files/mimic_index_frontal_valid.json"
    data.test_index_file="index_files/mimic_index_frontal_test.json"
    
    #processor configurations
    config.processor = processor = ml_collections.ConfigDict()
    processor.pretrained_model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor.use_fast= True
    processor.min_pixels= 496*496
    processor.max_pixels= 512*512
    
    #model configurations
    config.model = model = ml_collections.ConfigDict()
    model.pretrained_model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    #peft configurations
    config.peft = peft = ml_collections.ConfigDict()
    peft.USE_LORA = True
    peft.USE_QLORA = False
    peft.lora_alpha = 256
    peft.lora_r = 128
    peft.lora_dropout = 0.1
    peft.lora_bias = "none"
    peft.task_type = "CAUSAL_LM"
    
    #sft configurations
    config.sft_config = sft_config = ml_collections.ConfigDict()
    
    #inference
    peft.lora_adapter_path ="output_3b/final_model"
    peft.trained_model_path = "/home/arpanp/Fed_Vision_Language/output_3_cleaned/checkpoint-16500"
    
    return config
    
