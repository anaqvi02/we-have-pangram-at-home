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
        
        # Move to device
        self.to(self.config.DEVICE)
        
        # Experimental: torch.compile for Linux/CUDA speedup
        if self.config.DEVICE == "cuda" and hasattr(torch, "compile"):
            print("🚀 Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}. Proceeding without compilation.")
        
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
        import os
        # Graceful handling for local paths
        if not os.path.exists(path):
            # If it looks like a path (starts with / or ./), raise clear error
            if path.startswith("/") or path.startswith("./"):
                raise FileNotFoundError(f"Model path not found: {path}\nDid you finish training?")
            else:
                print(f"Path not found locally, assuming Hugging Face Hub ID: {path}")
        
        # We can just use standard HF loading for inference
        # But for training/fine-tuning we instantiate our class
        instance = cls(model_name=path)
        return instance

def get_optimizer(model, learning_rate=Config.LEARNING_RATE):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)
