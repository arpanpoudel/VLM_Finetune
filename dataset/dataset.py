from torch.utils.data import Dataset
import os
from PIL import Image
import json
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor
import random
import torch
class ChestDataset(Dataset):
    def __init__(self, index_path, transform=None, max_images=1, data_percent=10, seed=42):
        with open(index_path, "r") as f:
            self.samples = json.load(f)
        self.transform = transform
        self.max_images = max_images
        random.seed(seed)
        random.shuffle(self.samples)
        total_samples = len(self.samples)
        subset_size = int(total_samples * data_percent / 100)
        self.samples = self.samples[:subset_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = []
        for img_path in sample["image_paths"][:self.max_images]:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        with open(sample["text_path"], "r", encoding="utf-8") as f:
            text = f.read().strip()
        return {
            "image":images[0], "text": text}

class QwenDataCollator:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        self.system_message ="""You are an expert radiologist trained in interpreting chest X-rays. Given radiographic image of the chest, generate a detailed radiology report."""
    def format_data(self, sample):
        return [
            {
                "role":"system",
                "content" : [{"type": "text", "text": self.system_message}]
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
    
    def __call__(self, examples):
        # Extract images and texts from the batch
        examples = [self.format_data(sample) for sample in examples]
        texts =[
        self.processor.apply_chat_template(example, tokenize=False) for example in examples
        ]
        image_inputs = [process_vision_info(sample)[0] for sample in examples]
        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )
        #extract only assistant content
        input_ids_lists = batch['input_ids'].tolist()
        assert len(examples) == len(input_ids_lists)
        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            labels_list.append(label_ids)
        
        labels = torch.tensor(labels_list, dtype= torch.int64)
        batch["labels"] = labels  # Add labels to the batch

        return batch
 
def find_assistant_content_sublist_indexes(l):
    '''
    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))