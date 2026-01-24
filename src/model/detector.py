import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2TokenizerFast
from src.config import Config

class PangramDetector(torch.nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, num_labels=2):
        super().__init__()
        self.config = Config
        print(f"Initializing {model_name} on {self.config.DEVICE}...")
        
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
        
        # Move to MPS/device
        self.to(self.config.DEVICE)
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    @classmethod
    def load(cls, path):
        # We can just use standard HF loading for inference
        # But for training/fine-tuning we instantiate our class
        instance = cls(model_name=path)
        return instance

def get_optimizer(model, learning_rate=Config.LEARNING_RATE):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)
